# %% [markdown]
# # Assignment: Natural Language Inference with BERT and LLM Comparison
# 
# ## Overview
# 
# In this assignment, you are expected to design and implement a small experimental Natural Language Processing (NLP) pipeline using the Python programming language. Unlike strictly guided projects, this assignment allows flexibility in your experimental design and implementation choices.
# 
# You will use the **SNLI (Stanford Natural Language Inference)** dataset available on Hugging Face:
# 
# https://huggingface.co/datasets/stanfordnlp/snli
# 
# The dataset consists of sentence pairs:
# 
# * **Premise**
# * **Hypothesis**
# 
# Your task is to determine the relationship between these two sentences using one of the following labels:
# 
# * **Entailment:** The hypothesis can logically be inferred from the premise.
# * **Contradiction:** The hypothesis contradicts the premise.
# * **Neutral:** The hypothesis is neither supported nor contradicted by the premise.
# 
# The main objective is to:
# 
# 1. Fine-tune a BERT-based model using the SNLI training set.
# 2. Use a ready-made Large Language Model (LLM) for inference.
# 3. Compare the performances and outputs of both approaches.
# 
# ## Task 1: Data Exploration, Analysis and Preprocessing
# 
# In this stage, you should prepare the dataset using the Python `datasets` library. You are expected to explore and understand the dataset through analytical outputs and visualizations generated with code.
# 
# Possible analyses include (but are not limited to):
# 
# * Sentence length distributions
# * Total word counts
# * Stopword ratios
# * Label distributions
# * Vocabulary statistics
# * Premise vs. hypothesis length comparisons
# * Class imbalance analysis
# 
# You may include tables, plots, or statistical summaries to support your observations.
# 
# ### Preprocessing
# 
# You are free to apply any preprocessing steps you consider useful, such as:
# 
# * Lowercasing
# * Removing special characters
# * Stopword removal
# * Token normalization
# * Truncation or padding strategies
# * Filtering invalid samples
# 
# At the end of this stage, ensure that the whole dataset including validation and test splits is fully prepared for training phase.
# 
# > **Note:** This stage is intentionally flexible. If you choose not to perform any preprocessing or exploratory analysis, you must provide a reasonable explanation in markdown format discussing why you skipped these steps.

# %% [markdown]
# ---
# ### Task 1: Your Work Starts Here

# %% [markdown]
# ## 1.1 Environment Setup and Library Imports
# 
# We begin by installing and importing all necessary libraries.

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, ConvBertTokenizer
import umap
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

SEED = 55
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1.2 Data Loading and Initial Inspection
# 
# We load the SNLI dataset from Hugging Face. The dataset contains three splits: train, validation, and test. Each sample consists of a `premise`, a `hypothesis`, and a `label` (0=entailment, 1=neutral, 2=contradiction). Samples with `label == -1` are annotator-disagreement rows and must be filtered out before any analysis.

# %%
snli = load_dataset("stanfordnlp/snli")

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

print("=== Raw Dataset Sizes ===")
for split in ['train', 'validation', 'test']:
    print(f"  {split}: {len(snli[split]):,} samples")

# Filter out invalid labels (label == -1 indicates annotator disagreement)
snli_filtered = {}
for split in ['train', 'validation', 'test']:
    invalid_count = sum(1 for ex in snli[split] if ex['label'] == -1)
    snli_filtered[split] = snli[split].filter(lambda x: x['label'] != -1)
    print(f"  {split}: removed {invalid_count} invalid samples -> {len(snli_filtered[split]):,} remaining")

train_df = pd.DataFrame(snli_filtered['train'])
val_df = pd.DataFrame(snli_filtered['validation'])
test_df = pd.DataFrame(snli_filtered['test'])

print(f"\nFinal sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(train_df.head())

# %% [markdown]
# ## 1.3 Label Distribution and Class Imbalance Analysis
# 
# Understanding the class distribution is critical. If one label dominates, the model may develop a bias toward the majority class. We examine all three splits to verify whether the dataset is balanced.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, df) in zip(axes, [('Train', train_df), ('Validation', val_df), ('Test', test_df)]):
    counts = df['label'].value_counts().sort_index()
    bars = ax.bar([label_map[i] for i in counts.index], counts.values)
    ax.set_title(f'{name} Set Label Distribution')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Class imbalance ratio
print("=== Class Imbalance Ratios (Train) ===")
train_counts = train_df['label'].value_counts().sort_index()
total = train_counts.sum()
for idx, count in train_counts.items():
    print(f"  {label_map[idx]}: {count:,} ({count/total*100:.2f}%)")

imbalance_ratio = train_counts.max() / train_counts.min()
print(f"\nMax/Min class ratio: {imbalance_ratio:.4f}")
print("The dataset is approximately balanced." if imbalance_ratio < 1.5 else "WARNING: Notable class imbalance detected.")

# %% [markdown]
# [PLACEHOLDER: Interpretation of the label distribution and class imbalance analysis will be added here after the cell output is generated.]

# %% [markdown]
# ## 1.4 Sentence Length Distributions
# 
# We compute word-level lengths for both premise and hypothesis sentences. This reveals the typical input size the model will encounter and helps us set a reasonable `max_length` for tokenization later.

# %%
train_df['premise_len'] = train_df['premise'].apply(lambda x: len(x.split()))
train_df['hypothesis_len'] = train_df['hypothesis'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['premise_len'], bins=50, alpha=0.7, edgecolor='black')
axes[0].set_title('Premise Word Length Distribution')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].axvline(train_df['premise_len'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['premise_len'].mean():.1f}")
axes[0].legend()

axes[1].hist(train_df['hypothesis_len'], bins=50, alpha=0.7, edgecolor='black')
axes[1].set_title('Hypothesis Word Length Distribution')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].axvline(train_df['hypothesis_len'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['hypothesis_len'].mean():.1f}")
axes[1].legend()

plt.tight_layout()
plt.show()

print("=== Sentence Length Statistics ===")
for col, name in [('premise_len', 'Premise'), ('hypothesis_len', 'Hypothesis')]:
    print(f"  {name}: mean={train_df[col].mean():.1f}, median={train_df[col].median():.1f}, "
          f"std={train_df[col].std():.1f}, max={train_df[col].max()}")

# %% [markdown]
# [PLACEHOLDER: Interpretation of the sentence length distributions will be added here after the cell output is generated.]

# %% [markdown]
# ## 1.5 Premise vs. Hypothesis Sentence Length Comparison (Mean)
# 
# We compare the average word counts of premise and hypothesis sentences broken down by class. This reveals whether certain labels are associated with systematically shorter or longer sentences.

# %%
mean_lengths = train_df.groupby('label')[['premise_len', 'hypothesis_len']].mean()
mean_lengths.index = [label_map[i] for i in mean_lengths.index]

mean_lengths.plot(kind='bar', figsize=(8, 5), edgecolor='black')
plt.title('Mean Sentence Length by Class')
plt.xlabel('Label')
plt.ylabel('Mean Word Count')
plt.legend(['Premise', 'Hypothesis'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("=== Mean Lengths by Class ===")
print(mean_lengths.round(2).to_string())

# %% [markdown]
# [PLACEHOLDER: Interpretation of the premise vs. hypothesis length comparison will be added here after the cell output is generated.]

# %% [markdown]
# ## 1.6 Vocabulary Statistics
# 
# We analyze the total vocabulary size, the most common words, and how the vocabulary is distributed across premises and hypotheses. This provides insight into lexical diversity and potential redundancy.

# %%
all_premise_words = ' '.join(train_df['premise'].values).lower().split()
all_hypothesis_words = ' '.join(train_df['hypothesis'].values).lower().split()

premise_vocab = set(all_premise_words)
hypothesis_vocab = set(all_hypothesis_words)
combined_vocab = premise_vocab | hypothesis_vocab
shared_vocab = premise_vocab & hypothesis_vocab

print("=== Vocabulary Statistics (Train Set) ===")
print(f"  Total word tokens (Premise):    {len(all_premise_words):,}")
print(f"  Total word tokens (Hypothesis): {len(all_hypothesis_words):,}")
print(f"  Unique words (Premise):         {len(premise_vocab):,}")
print(f"  Unique words (Hypothesis):      {len(hypothesis_vocab):,}")
print(f"  Combined vocabulary size:        {len(combined_vocab):,}")
print(f"  Shared vocabulary size:          {len(shared_vocab):,}")
print(f"  Overlap ratio:                   {len(shared_vocab)/len(combined_vocab)*100:.2f}%")

# Top 20 most frequent words

meaningful_words = [word for word in (all_premise_words + all_hypothesis_words) 
                    if word not in stop_words and word.isalpha()]

combined_counter = Counter(meaningful_words)
top20 = combined_counter.most_common(20)
words, counts = zip(*top20)

plt.figure(figsize=(12, 4))
plt.bar(words, counts, edgecolor='black')
plt.title('Top 20 Most Frequent Words (Train)')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.7 Out-of-Vocabulary (OOV) Analysis
# 
# We use the ConvBERT tokenizer to check how many words in the dataset are out-of-vocabulary (i.e., not present as whole tokens in the tokenizer's vocabulary and thus get split into subword pieces). A high OOV rate signals that contraction expansion and lowercasing are important preprocessing steps.

# %%
convbert_tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')
unk_token_id = convbert_tokenizer.unk_token_id

def compute_oov_rate(texts, tokenizer, unk_id):
    """
    Computes the true Out-Of-Vocabulary (OOV) rate at the token level.
    This reflects exactly what the ConvBERT model 'sees' after WordPiece tokenization.
    
    @param texts: iterable of sentence strings
    @param tokenizer: HuggingFace tokenizer instance
    @param unk_id: integer id of the [UNK] token
    @return: tuple of (oov_rate, unk_token_count, total_token_count)
    """
    total_tokens, unk_tokens = 0, 0
    
    for text in texts:
        token_ids = tokenizer.encode(text.lower(), add_special_tokens=False)
        total_tokens += len(token_ids)
        unk_tokens += token_ids.count(unk_id)
    oov_rate = (unk_tokens / total_tokens) if total_tokens > 0 else 0
    return oov_rate, unk_tokens, total_tokens

for name, texts in [("Premise", train_df['premise']), ("Hypothesis", train_df['hypothesis'])]:
    rate, oov_c, total_c = compute_oov_rate(texts, convbert_tokenizer, unk_token_id)
    print(f"  {name}: OOV rate = {rate*100:.4f}% ({oov_c:,} / {total_c:,} total tokens)")

# %% [markdown]
# ## 1.8 Lexical Overlap Analysis (Jaccard Similarity)
# 
# This is one of the most important analyses for NLI. The "Lexical Overlap Bias" is a well-documented trap: if premise and hypothesis share most of their words, the model might learn a shortcut — predict entailment simply by counting shared words — without understanding meaning.
# 
# We compute the Jaccard Similarity between the word sets of premise and hypothesis for every sample, then plot the distribution across classes using a violin plot.

# %%
def jaccard_similarity(text1, text2):
    """
    Computes the Jaccard Similarity between two sentences based on their word sets.

    @param text1: first sentence string
    @param text2: second sentence string
    @return: float in [0, 1] representing the ratio of shared words to total unique words
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

train_df['jaccard'] = train_df.apply(lambda r: jaccard_similarity(r['premise'], r['hypothesis']), axis=1)

plt.figure(figsize=(10, 6))
data_for_plot = [train_df[train_df['label'] == i]['jaccard'].values for i in range(3)]
parts = plt.violinplot(data_for_plot, positions=[0, 1, 2], showmeans=True, showmedians=True)
plt.xticks([0, 1, 2], ['Entailment', 'Neutral', 'Contradiction'])
plt.title('Lexical Overlap (Jaccard Similarity) Distribution by Class')
plt.xlabel('Label')
plt.ylabel('Jaccard Similarity')
plt.tight_layout()
plt.show()

print("=== Jaccard Similarity Statistics by Class ===")
for label_id in range(3):
    subset = train_df[train_df['label'] == label_id]['jaccard']
    print(f"  {label_map[label_id]:15s}: mean={subset.mean():.4f}, median={subset.median():.4f}, std={subset.std():.4f}")

# %% [markdown]
# [PLACEHOLDER: Interpretation of the lexical overlap analysis will be added here after the cell output is generated.]

# %% [markdown]
# ## 1.9 UMAP Spatial Clustering Visualization
# 
# We project a random subset of 5,000 premise–hypothesis pairs into a 2D space using UMAP. The embeddings are generated using the `all-MiniLM-L6-v2` model loaded **directly via the `transformers` library**.
# 
# **Rationale:** If entailment and contradiction samples do not form clearly separated clusters, it signals that the model will have difficulty distinguishing these classes purely from semantic representations, suggesting the task genuinely requires deeper reasoning.

# %%
# Load the embedding model directly via transformers 
embed_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
embed_model.eval()

def mean_pooling(model_output, attention_mask):
    """
    Applies mean pooling to token embeddings, masked by attention_mask.

    @param model_output: output from the transformer model (last_hidden_state at index 0)
    @param attention_mask: tensor indicating real tokens (1) vs padding (0)
    @return: sentence-level embedding tensor of shape (batch_size, hidden_dim)
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, batch_size=64):
    """
    Generates sentence embeddings for a list of texts using the loaded transformer model.

    @param texts: list of sentence strings
    @param batch_size: number of sentences processed per forward pass
    @return: numpy array of shape (len(texts), hidden_dim)
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = embed_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = embed_model(**encoded)
        embs = mean_pooling(outputs, encoded['attention_mask'])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embeddings.append(embs.cpu().numpy())
    return np.vstack(all_embeddings)

# Sample 5000 rows
sample_idx = np.random.choice(len(train_df), size=5000, replace=False)
sample_df = train_df.iloc[sample_idx].reset_index(drop=True)

# Combine premise and hypothesis for joint embedding
combined_texts = (sample_df['premise'] + " [SEP] " + sample_df['hypothesis']).tolist()

print("Generating embeddings for 5,000 samples...")
embeddings = get_embeddings(combined_texts)
print(f"Embedding shape: {embeddings.shape}")

# UMAP projection
print("Running UMAP projection...")
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
umap_result = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
colors = ['#2196F3', '#FF9800', '#F44336']
for label_id in range(3):
    mask = sample_df['label'].values == label_id
    plt.scatter(umap_result[mask, 0], umap_result[mask, 1],
                c=colors[label_id], label=label_map[label_id], alpha=0.4, s=8)
plt.title('UMAP Projection of Premise-Hypothesis Pairs (n=5,000)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(markerscale=3)
plt.tight_layout()
plt.show()

# Cleanup GPU memory
del embed_model, embed_tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# [PLACEHOLDER: Interpretation of the UMAP visualization will be added here after the cell output is generated.]

# %% [markdown]
# ## 1.10 Percentile Token Length Calculation
# 
# We tokenize all premise–hypothesis pairs using the ConvBERT tokenizer and compute the token length distribution. The chosen percentile (e.g., 95th or 99th) will serve as the `max_length` hyperparameter for truncation during training. This avoids wasting memory on excessively long sequences while preserving the vast majority of input information.
# 
# **Important:** We define the helper function here but execute the actual computation **after preprocessing** (Section 1.11). This ensures `max_length` reflects the true tokenized lengths the model will encounter (e.g., after contraction expansion, `won't` becomes `will not`, adding tokens).

# %%
def compute_pair_token_lengths(premises, hypotheses, tokenizer):
    """
    Tokenizes premise-hypothesis pairs using batch encoding and returns token counts.

    @param premises: list of premise strings
    @param hypotheses: list of hypothesis strings
    @param tokenizer: HuggingFace tokenizer instance
    @return: numpy array of token lengths for each sample
    """
    encoded = tokenizer(premises, hypotheses, add_special_tokens=True)
    return np.array([len(ids) for ids in encoded['input_ids']])

print("Function defined. Token length computation will run after preprocessing (Section 1.11).")

# %% [markdown]
# ## 1.11 Preprocessing Pipeline
# 
# Our preprocessing strategy is guided by the following principles:
# 
# **What we DO:**
# - **Lowercasing:** Reduces vocabulary size and ensures case-invariant matching.
# - **Contraction Expansion:** Converts contractions (e.g., `won't` → `will not`) to their full forms, reducing OOV tokens for the subword tokenizer.
# - **Filter Invalid Samples:** Already done — removed `label == -1` rows.
# - **Delete Duplicates:** Removes exact duplicate premise-hypothesis-label triples to prevent data leakage and memorization.
# 
# **What we deliberately DO NOT do:**
# - **Stopword Removal:** Preserving stopwords is critical for NLI. Words like "not", "no", "never" are essential for detecting contradiction. Removing them would destroy the semantic signal the model needs.
# - **Punctuation Removal:** Punctuation carries contextual cues (e.g., question marks, commas) that help the tokenizer segment meaning correctly.
# 
# **What happens automatically :**
# - **Segment Encoding (Token Type IDs):** The HuggingFace tokenizer assigns `token_type_ids` of 0 and 1 when called with two sentences: `tokenizer(premise, hypothesis)`.
# - **Attention Masking:** The tokenizer automatically generates the `attention_mask` field.
# - **Dynamic Padding:** We will use `DataCollatorWithPadding` in the DataLoader to pad per-batch rather than globally.

# %%
CONTRACTION_MAP = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
    "let's": "let us", "ma'am": "madam", "o'clock": "of the clock",
    "y'all": "you all", "'cause": "because",
}

CONTRACTION_PATTERN = re.compile(
    '(' + '|'.join(re.escape(k) for k in sorted(CONTRACTION_MAP.keys(), key=len, reverse=True)) + ')',
    flags=re.IGNORECASE
)

def expand_contractions(text):
    """
    Expands English contractions in the input text to their full forms.

    @param text: input sentence string
    @return: sentence with contractions expanded
    """
    def replace_match(match):
        token = match.group(0)
        expanded = CONTRACTION_MAP.get(token.lower(), token)
        # Preserve original casing of first character
        if token[0].isupper():
            expanded = expanded[0].upper() + expanded[1:]
        return expanded
    return CONTRACTION_PATTERN.sub(replace_match, text)

def preprocess_text(text):
    """
    Applies the full preprocessing pipeline to a single text: lowercasing and contraction expansion.

    @param text: raw input sentence
    @return: cleaned sentence string
    """
    text = text.lower()
    text = expand_contractions(text)
    return text.strip()

# Verify contraction expansion
test_cases = ["I won't go", "She can't believe it", "They're here", "He's running"]
for tc in test_cases:
    print(f"  '{tc}' -> '{preprocess_text(tc)}'")

# %%
# Apply preprocessing to all splits
for df, name in [(train_df, 'train'), (val_df, 'validation'), (test_df, 'test')]:
    df['premise'] = df['premise'].apply(preprocess_text)
    df['hypothesis'] = df['hypothesis'].apply(preprocess_text)
    print(f"  {name}: preprocessing applied to {len(df):,} samples")

# Remove duplicates from training set
before_dedup = len(train_df)
train_df = train_df.drop_duplicates(subset=['premise', 'hypothesis', 'label']).reset_index(drop=True)
after_dedup = len(train_df)
print(f"\nDuplicate removal: {before_dedup:,} -> {after_dedup:,} (removed {before_dedup - after_dedup:,} duplicates)")

# Post-preprocessing OOV check
print("\n=== Post-Preprocessing OOV Rates ===")
for name, texts in [("Premise", train_df['premise']), ("Hypothesis", train_df['hypothesis'])]:
    rate, oov_c, total_c = compute_oov_rate(texts, convbert_tokenizer, unk_token_id)
    print(f"  {name}: OOV rate = {rate*100:.4f}% ({oov_c:,} / {total_c:,} words)")
print("\nContraction expansion should have reduced OOV rates compared to the pre-preprocessing values.")

# === Percentile Token Length (computed on PREPROCESSED data) ===
print("\nComputing token lengths on preprocessed train set...")
token_lengths = compute_pair_token_lengths(
    train_df['premise'].tolist(), train_df['hypothesis'].tolist(), convbert_tokenizer
)

percentiles = [50, 75, 90, 95, 98, 99, 100]
print("\n=== Token Length Percentiles ===")
for p in percentiles:
    val = int(np.percentile(token_lengths, p))
    print(f"  {p}th percentile: {val} tokens")

CHOSEN_PERCENTILE = 99
MAX_LENGTH = int(np.percentile(token_lengths, CHOSEN_PERCENTILE))
print(f"\n>> Selected max_length = {MAX_LENGTH} (based on {CHOSEN_PERCENTILE}th percentile)")
print(f"   This covers {(token_lengths <= MAX_LENGTH).sum() / len(token_lengths) * 100:.2f}% of all samples without truncation.")

plt.figure(figsize=(10, 4))
plt.hist(token_lengths, bins=60, edgecolor='black', alpha=0.7)
plt.axvline(MAX_LENGTH, color='red', linestyle='--', linewidth=2, label=f'{CHOSEN_PERCENTILE}th pctl = {MAX_LENGTH}')
plt.title('Token Length Distribution (Premise + Hypothesis, Post-Preprocessing)')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.12 Task 1 Summary
# 
# All required analyses have been completed:
# - **Label distribution** → approximately balanced across all three classes.
# - **Sentence length distributions** → premises are generally longer than hypotheses.
# - **Vocabulary statistics** → substantial overlap between premise and hypothesis vocabularies.
# - **OOV analysis** → contraction expansion reduced OOV rate.
# - **Lexical overlap (Jaccard)** → reveals the risk of lexical overlap bias per class.
# - **UMAP visualization** → shows the degree of class separability in semantic space.
# - **Percentile token length** → informs `max_length` hyperparameter.
# - **Class imbalance** → no significant imbalance detected.
# - **Preprocessing** → lowercasing, contraction expansion, duplicate removal applied; stopwords and punctuation deliberately preserved.

# %% [markdown]
# ### Task 1: Your Work Ends Here
# ---

# %% [markdown]
# ## Task 2: BERT Fine-Tuning
# 
# In this stage, you will fine-tune a BERT-based model using **only the training split** of the dataset.
# 
# The dataset already contains predefined train, validation, and test splits via the `datasets` library. You may use any transformer-based BERT architecture and any deep learning framework or library you prefer (e.g., Hugging Face Transformers, PyTorch, TensorFlow, etc.).
# 
# ### Requirements
# 
# You must clearly explain and demonstrate:
# 
# * How the two texts were combined
# * Tokenization strategy
# * Input formatting
# * Hyperparameter choices
# * Training setup
# * Validation strategy
# * Experimental observations
# 
# Include outputs, comments, and explanations throughout your notebook or report (markdown explanations are required).
# 
# > **Important:** The test set must **not** be used during training or validation.
# 
# ### Evaluation and Visualization
# 
# There will be three primary evaluation metrics you should consider:
# 
# * Confusion Matrix
# * Accuracy
# * Macro F1-Score
# 
# You are expected to use the **entire test split** of the dataset during the evaluation phase.
# 
# You must include visual outputs related to the training process, such as:
# 
# * Training/validation loss curves
# * Confusion Matrix, Accuracy, and Macro F1-Score plots
# 
# You are encouraged to interpret and discuss your results.

# %% [markdown]
# ---
# ### Task 2: Your Work Starts Here

# %% [markdown]
# ## 2.1 ConvBERT Model Setup and Layer Freezing Strategy
# 
# We use **ConvBERT** (`YituTech/conv-bert-base`) for sequence classification. ConvBERT replaces some of BERT's global self-attention heads with span-based dynamic convolutions, making it more efficient while retaining strong performance.
# 
# **Layer Freezing Strategy:** Instead of fine-tuning all 12 encoder layers, we freeze the embedding layer and the first 8 encoder layers (0–7), and only train the **last 4 encoder layers (8–11)** plus the classification head. This approach:
# - Preserves the general linguistic knowledge learned during pretraining in the lower layers.
# - Allows the upper layers to adapt to the NLI task.
# - Significantly reduces training time and GPU memory requirements.
# 
# **How the two texts are combined:** We pass both premise and hypothesis directly to the tokenizer as `tokenizer(premise, hypothesis)`. The tokenizer automatically:
# 1. Concatenates them with a `[SEP]` token: `[CLS] premise [SEP] hypothesis [SEP]`
# 2. Generates `token_type_ids` (0 for premise tokens, 1 for hypothesis tokens) — **Segment Encoding**
# 3. Generates `attention_mask` (1 for real tokens, 0 for padding) — **Attention Masking**
# 
# **Dynamic Padding:** We use `DataCollatorWithPadding` which pads each batch to the length of its longest sequence, rather than padding everything to `max_length` globally. This saves memory and speeds up training.
# 
# **Max Length:** Set to the value computed from the percentile token length analysis in Task 1.

# %%
from transformers import ConvBertForSequenceClassification, ConvBertTokenizer, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup
import transformers
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from tqdm.auto import tqdm
import time
transformers.logging.set_verbosity_error()
NUM_LABELS = 3
MODEL_NAME = 'YituTech/conv-bert-base'

tokenizer = ConvBertTokenizer.from_pretrained(MODEL_NAME)
model = ConvBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)

print(f"Model: {MODEL_NAME}")
print(f"Max length: {MAX_LENGTH}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Freeze embedding layer and encoder layers 0-7; train only layers 8-11 + classifier
for param in model.convbert.embeddings.parameters():
    param.requires_grad = False

for i in range(8):
    for param in model.convbert.encoder.layer[i].parameters():
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {frozen:,}")
print(f"Trainable ratio:      {trainable/(trainable+frozen)*100:.2f}%")

# Verify which layers are trainable
print("\n=== Layer Freeze Status ===")
print(f"  Embeddings: FROZEN")
for i in range(12):
    status = "TRAINABLE" if any(p.requires_grad for p in model.convbert.encoder.layer[i].parameters()) else "FROZEN"
    print(f"  Encoder Layer {i:2d}: {status}")
print(f"  Classifier:     TRAINABLE")

# %% [markdown]
# ## 2.2 Dataset Preparation and Tokenization
# 
# We create a PyTorch `Dataset` that tokenizes premise–hypothesis pairs on-the-fly. The tokenizer handles segment encoding and attention masking automatically. We do **not** pad to `max_length` here — that is deferred to the `DataCollatorWithPadding`.

# %%
class SNLIDataset(Dataset):
    """
    PyTorch Dataset wrapper for SNLI premise-hypothesis pairs.

    @param premises: list of premise strings
    @param hypotheses: list of hypothesis strings
    @param labels: list of integer labels (0, 1, 2)
    @param tokenizer: HuggingFace tokenizer instance
    @param max_length: maximum token sequence length for truncation
    """
    def __init__(self, premises, hypotheses, labels, tokenizer, max_length):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            truncation=True,
            max_length=self.max_length,
            # No padding here — DataCollatorWithPadding handles it per-batch
        )
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_subset_df, _ = train_test_split(
    train_df, 
    train_size=100000, 
    stratify=train_df['label'], 
    random_state=55
)


train_subset_df = train_subset_df.reset_index(drop=True)


train_dataset = SNLIDataset(
    train_subset_df['premise'].tolist(), train_subset_df['hypothesis'].tolist(),
    train_subset_df['label'].tolist(), tokenizer, MAX_LENGTH
)
val_dataset = SNLIDataset(
    val_df['premise'].tolist(), val_df['hypothesis'].tolist(),
    val_df['label'].tolist(), tokenizer, MAX_LENGTH
)
test_dataset = SNLIDataset(
    test_df['premise'].tolist(), test_df['hypothesis'].tolist(),
    test_df['label'].tolist(), tokenizer, MAX_LENGTH
)

print(f"Train dataset: {len(train_dataset):,} samples")
print(f"Val dataset:   {len(val_dataset):,} samples")
print(f"Test dataset:  {len(test_dataset):,} samples")

# Inspect a single tokenized sample
sample = train_dataset[0]
print(f"\nSample keys: {list(sample.keys())}")
print(f"  input_ids shape:      {sample['input_ids'].shape}")
print(f"  token_type_ids shape: {sample['token_type_ids'].shape}")
print(f"  attention_mask shape: {sample['attention_mask'].shape}")
print(f"  label: {sample['labels'].item()} ({label_map[sample['labels'].item()]})")
print(f"\nDecoded: {tokenizer.decode(sample['input_ids'])}")

# %% [markdown]
# ## 2.3 Hyperparameters and Training Setup
# 
# **Hyperparameter Choices:**
# - **Batch size = 32:** Standard choice balancing GPU memory usage and gradient stability.
# - **Learning rate = 2e-5:** Conservative rate typical for fine-tuning pretrained transformers; prevents catastrophic forgetting.
# - **Epochs = 3:** Sufficient for fine-tuning with a large dataset; more epochs risk overfitting.
# - **Warmup = 10% of total steps:** Gradually ramps up the learning rate to stabilize early training.
# - **Optimizer = AdamW:** Includes weight decay for regularization; the standard for transformer fine-tuning.
# - **Scheduler = Linear with warmup:** Decays the learning rate smoothly after the warmup period.
# 
# **Validation Strategy:** We evaluate on the full validation set at the end of every epoch and save the model with the best validation accuracy. The test set is never touched during training.

# %%
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_RATIO = 0.1

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.02
)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

print(f"Batch size:     {BATCH_SIZE}")
print(f"Learning rate:  {LEARNING_RATE}")
print(f"Epochs:         {EPOCHS}")
print(f"Total steps:    {total_steps:,}")
print(f"Warmup steps:   {warmup_steps:,}")
print(f"Train batches:  {len(train_loader):,}")
print(f"Val batches:    {len(val_loader):,}")

# %% [markdown]
# ## 2.4 Training Loop
# 
# We train the model with per-epoch validation. Training and validation losses are recorded for plotting loss curves. The best model (by validation accuracy) is saved.

# %%
def plot_live_metrics(train_losses, val_losses, val_accuracies):
    """
    Dynamically updates and displays the loss and accuracy curves during training.
    Clears the current output in the Jupyter/Colab cell to create a live-updating effect.

    @param train_losses: list of float, training loss values recorded at specific logging steps.
    @param val_losses: list of float, validation loss values recorded at the end of each epoch.
    @param val_accuracies: list of float, validation accuracy scores recorded at the end of each epoch.
    @return: None. Renders a matplotlib figure inline.
    """
    clear_output(wait=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    ax[0].plot(train_losses, label='Train Loss', marker='o', color='blue')
    if val_losses:
        ax[0].plot(val_losses, label='Validation Loss', marker='o', color='orange')
    ax[0].set_title('Live Loss Curve')
    ax[0].set_xlabel('Epochs (or Logging Steps)')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.6)

    if val_accuracies:
        ax[1].plot(val_accuracies, label='Validation Accuracy', marker='s', color='green')
        ax[1].set_title('Live Validation Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def evaluate(model, dataloader, device):
    """
    Evaluates the model on a given dataloader and returns loss, accuracy, and predictions.

    @param model: the ConvBERT model
    @param dataloader: PyTorch DataLoader for the evaluation set
    @param device: torch device (cpu or cuda)
    @return: dict with keys 'loss', 'accuracy', 'f1', 'preds', 'labels'
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return {'loss': avg_loss, 'accuracy': acc, 'f1': f1, 'preds': all_preds, 'labels': all_labels}

train_losses_history, val_losses_history = [], []
val_accuracies_history, val_f1s_history = [], []
best_val_acc = 0.0
log_step_interval = max(1, len(train_loader) // 10) 

print("Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    train_preds, train_labels = [], []
    start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        batch_preds = torch.argmax(outputs.logits, dim=-1)
        train_preds.extend(batch_preds.detach().cpu().numpy())
        train_labels.extend(batch['labels'].cpu().numpy())
        
        progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        if (step + 1) % log_step_interval == 0:
            current_avg_loss = epoch_loss / (step + 1)
            train_losses_history.append(current_avg_loss)
            plot_live_metrics(train_losses_history, val_losses_history, val_accuracies_history)

    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    
    val_result = evaluate(model, val_loader, device)
    val_losses_history.append(val_result['loss'])
    val_accuracies_history.append(val_result['accuracy'])
    val_f1s_history.append(val_result['f1'])
    
    val_class_report = classification_report(
        val_result['labels'], val_result['preds'], 
        target_names=['Entailment', 'Neutral', 'Contradiction'], output_dict=True
    )
    
    plot_live_metrics(train_losses_history, val_losses_history, val_accuracies_history)

    elapsed = time.time() - start_time
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} Summary ({elapsed:.0f}s) ---")
    print(f"  Train Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"  Val Loss:   {val_result['loss']:.4f} | Val Acc:   {val_result['accuracy']:.4f} | Val F1:   {val_result['f1']:.4f}")
    print(f"  Val Class Acc -> Entailment: {val_class_report['Entailment']['recall']*100:.1f}% | Neutral: {val_class_report['Neutral']['recall']*100:.1f}% | Contradiction: {val_class_report['Contradiction']['recall']*100:.1f}%")

    if val_result['accuracy'] > best_val_acc:
        best_val_acc = val_result['accuracy']
        torch.save(model.state_dict(), 'best_convbert_nli.pt')
        print(f"  >> New best model saved! (Val Acc: {best_val_acc:.4f})")
    print()

print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")

# %% [markdown]
# ## 2.5 Training and Validation Loss Curves

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Get the actual lengths of the history lists
val_epochs = len(val_losses_history)
train_steps = len(train_losses_history)

# Integer X-axis for validation metrics (1, 2, 3... epochs)
val_x = range(1, val_epochs + 1)

epoch_multiplier = val_epochs if val_epochs > 0 else (train_steps / 5)
train_x = [i * (epoch_multiplier / train_steps) for i in range(1, train_steps + 1)]

# 1. Loss Plot
axes[0].plot(train_x, train_losses_history, 'o-', label='Train Loss', color='blue', alpha=0.7)
if val_epochs > 0:
    axes[0].plot(val_x, val_losses_history, 's-', label='Val Loss', color='orange', linewidth=2)
axes[0].set_title('Training vs Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_xticks(list(range(1, max(2, val_epochs + 1))))
axes[0].grid(True, linestyle='--', alpha=0.6)

# 2. Accuracy Plot
if val_epochs > 0:
    axes[1].plot(val_x, val_accuracies_history, 's-', color='green', label='Val Accuracy', linewidth=2)
axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_xticks(list(val_x))
axes[1].grid(True, linestyle='--', alpha=0.6)

# 3. F1-Score Plot
if val_epochs > 0:
    axes[2].plot(val_x, val_f1s_history, 's-', color='red', label='Val Macro F1', linewidth=2)
axes[2].set_title('Validation Macro F1-Score')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Macro F1')
axes[2].legend()
axes[2].set_xticks(list(val_x))
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# [PLACEHOLDER: Interpretation of the training curves will be added here after the cell output is generated.]

# %% [markdown]
# ## 2.6 Test Set Evaluation
# 
# We load the best model and evaluate on the **entire test set**. This produces the confusion matrix, accuracy, and macro F1-score.

# %%
# Load the best model
model.load_state_dict(torch.load('best_convbert_nli.pt', map_location=device))
print("Best model loaded.\n")

test_result = evaluate(model, test_loader, device)

print("=== Test Set Results (Full) ===")
print(f"  Accuracy:     {test_result['accuracy']:.4f}")
print(f"  Macro F1:     {test_result['f1']:.4f}")
print(f"  Loss:         {test_result['loss']:.4f}")
print(f"\n{classification_report(test_result['labels'], test_result['preds'], target_names=['entailment', 'neutral', 'contradiction'])}")

# Confusion Matrix
cm = confusion_matrix(test_result['labels'], test_result['preds'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['entailment', 'neutral', 'contradiction'],
            yticklabels=['entailment', 'neutral', 'contradiction'])
plt.title(f'ConvBERT Confusion Matrix (Test Set)\nAccuracy={test_result["accuracy"]:.4f} | Macro F1={test_result["f1"]:.4f}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# %% [markdown]
# [PLACEHOLDER: Interpretation of the ConvBERT test set results, confusion matrix, and classification report will be added here after the cell output is generated.]

# %% [markdown]
# ## 2.7 Evaluate ConvBERT on the First 1,000 Test Samples
# 
# For a fair comparison with the LLM in Task 4, we also evaluate ConvBERT on the same 1,000-sample subset that will be used for llm inference.

# %%
# Prepare the 1000-sample subset for later comparison
test_1k_df = test_df.head(1000).reset_index(drop=True)
test_1k_dataset = SNLIDataset(
    test_1k_df['premise'].tolist(), test_1k_df['hypothesis'].tolist(),
    test_1k_df['label'].tolist(), tokenizer, MAX_LENGTH
)
test_1k_loader = DataLoader(test_1k_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

bert_1k_result = evaluate(model, test_1k_loader, device)

print("=== ConvBERT on First 1,000 Test Samples ===")
print(f"  Accuracy:  {bert_1k_result['accuracy']:.4f}")
print(f"  Macro F1:  {bert_1k_result['f1']:.4f}")

# Store for Task 4 comparison
bert_1k_preds = bert_1k_result['preds']
bert_1k_labels = bert_1k_result['labels']

# %% [markdown]
# ### Task 2: Your Work Ends Here
# ---

# %% [markdown]
# ## Task 3: Evaluating LLMs on NLI
# 
# In this task, the objective is to evaluate how powerful general-purpose Large Language Models (LLMs) perform on the Natural Language Inference (NLI) task **without task-specific fine-tuning**, and to compare their behavior against the fine-tuned BERT model developed in Task 1.
# 
# Unlike BERT, which produces deterministic class predictions after supervised fine-tuning, LLMs rely entirely on prompting and instruction-following capabilities. Therefore, prompt engineering and output validation become critical components of the experimental pipeline.
# 
# You may choose one or both of the following approaches:
# 
# ### Option A: Local Open-Source LLMs
# 
# Use a lightweight or quantized open-source LLM locally without performing additional training or fine-tuning. Possible examples include:
# 
# * Gemma
# * Llama
# * Mistral
# * Phi
# * Qwen
# 
# You may run these models using frameworks such as Hugging Face Transformers, llama.cpp, Ollama, or vLLM. This option is recommended for those who are interested in local deployment, hardware efficiency, or fully offline experimentation.
# 
# ### Option B: Cloud API-Based Inference (Recommended)
# 
# Use the **Google Gemini API** to access Gemma 3 or Gemma 4 through Google AI Studio. This option simplifies deployment and typically provides stronger instruction-following performance.
# 
# > **Important:** Free-tier usage is limited to approximately **1,500 requests per day per model**. Plan your experiments carefully before running inference on the full test set. You may refer to `Google AI Studio` for more information.
# 
# ### Prompt Engineering and Analysis
# 
# Since the LLM is not fine-tuned specifically for NLI, the prompt acts as the primary mechanism for guiding the model’s reasoning process. You must design a clear and robust prompt that:
# 
# * Explains the NLI task
# * Defines the three target labels (Entailment, Neutral, Contradiction)
# * Specifies the expected output format
# * Minimizes ambiguous or verbose responses
# 
# > **Note:** You are **strictly forbidden** from including any examples from the **test set** inside your prompt.
# 
# However, you may:
# 
# * Use observations from the training set
# * Use few-shot examples sampled only from the training split
# * Incorporate heuristics discovered during exploratory data analysis
# 
# ### Output Formatting
# 
# You must instruct the LLM to return outputs in a strict and parsable format. Recommended formats include:
# 
# \```text
# 
# Entailment
# 
# \```
# 
# or
# 
# \```json
# 
# {"label": "Neutral"}
# 
# \```
# 
# The simpler the output format, the easier it will be to parse and evaluate automatically.
# 
# ### Inference on the Test Set
# 
# Run inference on the same test split used during BERT evaluation.
# 
# > **Note:** Due to the computational requirements of LLMs and/or free API rate limitations, you may optionally perform inference on **the first subset of 1,000 test samples** instead of using the entire test set.
# 
# Because LLMs are non-deterministic and may fail to follow instructions consistently, you must implement a robust parsing and validation pipeline. Your implementation should properly handle:
# 
# * **Valid Predictions:** Outputs that can be mapped correctly to one of the three labels (Entailment, Neutral, Contradiction).
# * **Invalid Predictions:** Cases where the model produces ambiguous outputs, hallucinates additional explanations, returns malformed JSON, generates labels outside the allowed classes, or ignores formatting instructions. You should explicitly track and report invalid outputs during evaluation.

# %% [markdown]
# ---
# ### Task 3: Your Work Starts Here

# %% [markdown]
# ## 3.1 LLM Inference Setup — LLM via Google AI Studio API
# 
# We use **Gemini 3.1 Flash Lite** through the Google Gemini API for Few-shot NLI classification. The key design decisions:
# 
# - **Temperature = 0.05:** Very low temperature to maximize deterministic, consistent outputs.
# - **Rate limiting = 15 RPM:** We enforce a strict rate limit of 15 requests per minute to comply with API constraints.
# - **Structured output:** We request `application/json` responses with a strict schema forcing the model to return `{"label": "entailment"|"neutral"|"contradiction"}`.
# - **Subset = First 1,000 test samples:** Matching the subset used for ConvBERT comparison.
# 
# > **Note:** Replace `YOUR_API_KEY` with your actual Google AI Studio API key before running.

# %%
import google.generativeai as genai
import time
import json

# API key and model configuration
GEMINI_API_KEY = "GEMINI3_1_flash_lite_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.05
RPM_LIMIT = 10
BATCH_SIZE_LLM = 10
DELAY_BETWEEN_REQUESTS = 60.0 / RPM_LIMIT

# Structured output: wraps the array inside {"results": [...]}
# because response_mime_type requires a top-level object, not a bare array
api_model = genai.GenerativeModel(
    MODEL,
    generation_config=genai.types.GenerationConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "label": {
                                "type": "string",
                                "enum": ["entailment", "neutral", "contradiction"]
                            }
                        },
                        "required": ["id", "label"]
                    }
                }
            },
            "required": ["results"]
        },
    )
)

print(f"Model:       {MODEL}")
print(f"Temperature: {TEMPERATURE}")
print(f"RPM Limit:   {RPM_LIMIT}")
print(f"Batch size:  {BATCH_SIZE_LLM} pairs per request")
total_requests = (1000 + BATCH_SIZE_LLM - 1) // BATCH_SIZE_LLM
print(f"Total API calls: ~{total_requests}")
print(f"Estimated time:  ~{total_requests * DELAY_BETWEEN_REQUESTS / 60:.1f} minutes")

# %% [markdown]
# ## 3.2 Prompt Design
# 
# The prompt is designed to be clear, minimal, and unambiguous. We:
# 1. Define the NLI task concisely.
# 2. Specify the three labels with brief definitions.
# 3. Provide exactly two few-shot examples from the **training set** (not the test set).
# 4. Instruct the model to return only JSON.
# 
# The structured output schema already constrains the response, but the prompt reinforcement further reduces invalid outputs.

# %%
BATCH_PROMPT_TEMPLATE = """You are a Natural Language Inference (NLI) classifier. For each numbered pair, determine the relationship between the Premise and Hypothesis.

Labels:
- "entailment": The hypothesis logically follows from the premise.
- "neutral": The hypothesis is possible but not guaranteed by the premise.
- "contradiction": The hypothesis directly contradicts the premise.

Example:

Pair 1:
Premise: "A person is riding a horse."
Hypothesis: "A person is outdoors."

Pair 2:
Premise: "A woman is reading a book at the library."
Hypothesis: "A woman is reading a mystery novel."

Pair 3:
Premise: "A man is playing guitar on stage."
Hypothesis: "A man is sleeping in bed."

Answer:
{{"results": [{{"id": 1, "label": "entailment"}}, {{"id": 2, "label": "neutral"}}, {{"id": 3, "label": "contradiction"}}]}}

Now classify:

{pairs_block}"""


def build_pairs_block(premises, hypotheses):
    """
    Builds a numbered list of premise-hypothesis pairs for the batch prompt.

    @param premises: list of premise strings
    @param hypotheses: list of hypothesis strings
    @return: formatted string block with numbered pairs
    """
    lines = []
    for i, (p, h) in enumerate(zip(premises, hypotheses), 1):
        lines.append(f'Pair {i}:\nPremise: "{p}"\nHypothesis: "{h}"')
    return "\n\n".join(lines)


def classify_batch(premises, hypotheses, model_instance, max_retries=3, debug=False):
    """
    Sends a batch of NLI pairs in a single API call with enforced JSON schema.
    The API forces the model to return {"results": [{"id": N, "label": "..."}]}
    so no free-text analysis can leak through.

    @param premises: list of premise strings (batch)
    @param hypotheses: list of hypothesis strings (batch)
    @param model_instance: initialized GenerativeModel instance
    @param max_retries: number of retry attempts on transient failures
    @param debug: if True, prints raw response for troubleshooting
    @return: list of predicted label strings (or None for unparseable entries)
    """
    n = len(premises)
    pairs_block = build_pairs_block(premises, hypotheses)
    prompt = BATCH_PROMPT_TEMPLATE.format(pairs_block=pairs_block)
    valid_labels = {"entailment", "neutral", "contradiction"}

    for attempt in range(max_retries + 1):
        try:
            response = model_instance.generate_content(prompt)

            # Safe text extraction
            raw_text = None
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    raw_text = candidate.content.parts[0].text

            if not raw_text:
                if debug:
                    print(f"    [DEBUG] Empty response on attempt {attempt+1}")
                if attempt < max_retries:
                    time.sleep(20)
                    continue
                return [None] * n

            if debug:
                print(f"    [DEBUG] Raw response: {raw_text[:300]}")

            parsed = json.loads(raw_text)
            items = parsed.get("results", [])

            if isinstance(items, list) and len(items) > 0:
                results = []
                for item in items:
                    label = item.get("label", "").lower().strip() if isinstance(item, dict) else None
                    results.append(label if label in valid_labels else None)
                while len(results) < n:
                    results.append(None)
                return results[:n]
            else:
                if debug:
                    print(f"    [DEBUG] 'results' key missing or empty")
                if attempt < max_retries:
                    time.sleep(20)
                    continue
                return [None] * n

        except Exception as e:
            if debug:
                print(f"    [DEBUG] Exception on attempt {attempt+1}: {e}")
            if attempt < max_retries:
                time.sleep(20)
                continue
            return [None] * n

    return [None] * n

# %% [markdown]
# ## 3.3 Rate-Limited Inference on 1,000 Test Samples
# 
# We process the first 1,000 test samples with strict rate limiting. Each response is parsed from JSON, and any failures are tracked as invalid predictions.

# %%
test_1k_premises = test_1k_df['premise'].tolist()
test_1k_hypotheses = test_1k_df['hypothesis'].tolist()
test_1k_true = test_1k_df['label'].tolist()

llm_predictions = []
llm_true_labels = []

total_batches = (len(test_1k_premises) + BATCH_SIZE_LLM - 1) // BATCH_SIZE_LLM

print(f"Starting batched {MODEL} inference: {len(test_1k_premises)} samples in {total_batches} batches...\n")
start_time = time.time()

for batch_idx in range(total_batches):
    start_i = batch_idx * BATCH_SIZE_LLM
    end_i = min(start_i + BATCH_SIZE_LLM, len(test_1k_premises))

    batch_premises = test_1k_premises[start_i:end_i]
    batch_hypotheses = test_1k_hypotheses[start_i:end_i]
    batch_true = test_1k_true[start_i:end_i]
    start_time_batch = time.time()
    # Debuging first 2 batches to see raw responses
    use_debug = (batch_idx < 2)
    batch_preds = classify_batch(batch_premises, batch_hypotheses, api_model, debug=use_debug)

    llm_predictions.extend(batch_preds)
    llm_true_labels.extend(batch_true)
    response_time = time.time() - start_time_batch
    if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1 or batch_idx < 2:
        valid_so_far = sum(1 for p in llm_predictions if p is not None)
        elapsed = time.time() - start_time
        print(f"  Batch {batch_idx+1}/{total_batches} | "
              f"Samples: {len(llm_predictions)}/{len(test_1k_premises)} | "
              f"Valid: {valid_so_far}/{len(llm_predictions)} | "
              f"Elapsed: {elapsed/60:.1f}min")

    # Rate limiting
    if batch_idx < total_batches - 1:
        time.sleep(max(0, DELAY_BETWEEN_REQUESTS - response_time))

total_time = time.time() - start_time
print(f"\nInference complete in {total_time/60:.1f} minutes.")

valid_count = sum(1 for p in llm_predictions if p is not None)
invalid_count = sum(1 for p in llm_predictions if p is None)
print(f"Valid predictions:   {valid_count}/{len(llm_predictions)}")
print(f"Invalid predictions: {invalid_count}/{len(llm_predictions)}")

# %%
# Convert to integer labels for Task 4
label_to_int = {"entailment": 0, "neutral": 1, "contradiction": 2}

llm_pred_ints = []
for pred in llm_predictions:
    if pred is not None:
        llm_pred_ints.append(label_to_int[pred])
    else:
        llm_pred_ints.append(-1)

invalid_indices = [i for i, p in enumerate(llm_predictions) if p is None]
if invalid_indices:
    print(f"\n=== {len(invalid_indices)} Invalid Predictions (showing first 5) ===")
    for idx in invalid_indices[:5]:
        print(f"  Index {idx}: pred=None")
else:
    print("No invalid predictions — batch structured output worked perfectly.")

# %% [markdown]
# [PLACEHOLDER: Interpretation of the Gemini 3.1 flash lite inference results, including valid/invalid prediction rates, will be added here after the cell output is generated.]

# %% [markdown]
# ### Task 3: Your Work Ends Here
# ---

# %% [markdown]
# ## Task 4: Visualization and Comparison
# 
# ### 4.1 LLM Metric Calculation
# 
# Unlike BERT classifiers, LLMs may generate invalid or unparsable responses. Therefore, additional evaluation metrics are required. You must calculate the following:
# 
# **Overall Accuracy:**
# 
# 
# $$\frac{\text{Correct Predictions}}{\text{Total Samples}}$$
# 
# 
# *(This metric penalizes invalid responses.)*
# 
# **Invalid Rate:**
# 
# 
# $$\frac{\text{Invalid Predictions}}{\text{Total Samples}}$$
# 
# 
# *(This measures how often the LLM fails to follow instructions correctly.)*
# 
# **Adjusted Accuracy:**
# 
# 
# $$\frac{\text{Correct Predictions}}{\text{Valid Predictions Only}}$$
# 
# 
# *(This metric evaluates the model only on successfully parsed predictions.)*
# 
# **Macro F1-Score:**
# Compute the macro-averaged F1-score over the entire test subset. You must NOT drop or exclude invalid, unparsable, or missing predictions from your dataset. Any such output must be strictly treated as an incorrect prediction (misclassification) during your metric calculations.
# 
# ### 4.2 Visual Analysis
# 
# Generate the following visualizations:
# 
# **Side-by-Side Confusion Matrices**
# 
# > Use the same 1000 test split and evaluate your fine-tuned BERT model and LLM pipeline.
# 
# Display confusion matrices for:
# 
# 1. Fine-tuned BERT
# 2. LLM-based inference
# 
# You may include an additional category such as `"Invalid"` or `"N/A"` to visualize LLM formatting failures or parsing issues.
# 
# **Performance Comparison Bar Chart**
# Create a bar chart comparing:
# 
# * Accuracy
# * Adjusted Accuracy
# * Macro F1-Score
# * Invalid Rate
# 
# across both approaches. You are encouraged to include clear labels, legends, and interpretations.

# %% [markdown]
# ---
# ### Task 4: Your Work Starts Here

# %% [markdown]
# ## 4.1 LLM Metric Calculation
# 
# We compute the four required metrics for the LLM:
# - **Overall Accuracy** = Correct / Total (penalizes invalids)
# - **Invalid Rate** = Invalid / Total
# - **Adjusted Accuracy** = Correct / Valid Only
# - **Macro F1-Score** = Treating invalids as misclassifications (not excluded)

# %%
# Use exactly what was processed — no assumptions about 1000
total_samples = len(llm_pred_ints)
print(f"Total samples processed: {total_samples}")

valid_mask = [p != -1 for p in llm_pred_ints]
valid_preds = [p for p, m in zip(llm_pred_ints, valid_mask) if m]
valid_labels = [l for l, m in zip(llm_true_labels, valid_mask) if m]

correct_valid = sum(1 for p, l in zip(valid_preds, valid_labels) if p == l)

# Metrics
overall_accuracy = correct_valid / total_samples
invalid_rate = sum(1 for m in valid_mask if not m) / total_samples
adjusted_accuracy = correct_valid / len(valid_preds) if valid_preds else 0.0

# Macro F1: pred=-1 penalizes recall without injecting fake false positives
macro_f1_llm = f1_score(llm_true_labels, llm_pred_ints, average='macro', labels=[0, 1, 2])

print(f"\n=== Gemini 3.1 Flash Lite Metrics ({total_samples} Test Samples) ===")
print(f"  Overall Accuracy:  {overall_accuracy:.4f}")
print(f"  Invalid Rate:      {invalid_rate:.4f} ({sum(1 for m in valid_mask if not m)}/{total_samples})")
print(f"  Adjusted Accuracy: {adjusted_accuracy:.4f}")
print(f"  Macro F1-Score:    {macro_f1_llm:.4f}")


bert_acc_sub = accuracy_score(bert_1k_labels, bert_1k_preds)
bert_f1_sub = f1_score(bert_1k_labels, bert_1k_preds, average='macro')

print(f"\n=== ConvBERT Metrics (Same {total_samples} Test Samples) ===")
print(f"  Overall Accuracy:     {bert_acc_sub:.4f}")
print(f"  Invalid Rate:         0.0000  (deterministic model)")
print(f"  Adjusted Accuracy:    {bert_acc_sub:.4f}  (same as accuracy, no invalids)")
print(f"  Macro F1-Score:       {bert_f1_sub:.4f}")

# %% [markdown]
# ## 4.2 Side-by-Side Confusion Matrices

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ConvBERT Confusion Matrix (1K subset)
cm_bert = confusion_matrix(bert_1k_labels, bert_1k_preds, labels=[0, 1, 2])
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['entailment', 'neutral', 'contradiction'],
            yticklabels=['entailment', 'neutral', 'contradiction'])
axes[0].set_title(f'ConvBERT ({total_samples} Subset)\nAcc={bert_acc_sub:.4f} | F1={bert_f1_sub:.4f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# LLM Confusion Matrix (with Invalid category on predicted axis)
llm_display_preds = []
for p in llm_pred_ints:
    llm_display_preds.append(p if p != -1 else 3)  # 3 = Invalid

cm_llm_full = confusion_matrix(llm_true_labels, llm_display_preds, labels=[0, 1, 2, 3])
cm_llm = cm_llm_full[:3, :]  # Keep only 3 true-label rows (Invalid never appears as a true label)

sns.heatmap(cm_llm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['entailment', 'neutral', 'contradiction', 'Invalid'],
            yticklabels=['entailment', 'neutral', 'contradiction'])
axes[1].set_title(f'Gemini 3.1 Flash Lite ({total_samples} Subset)\nAcc={overall_accuracy:.4f} | F1={macro_f1_llm:.4f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True');

# %% [markdown]
# [PLACEHOLDER: Interpretation of the side-by-side confusion matrices will be added here after the cell output is generated.]

# %% [markdown]
# ## 4.3 Performance Comparison Bar Chart

# %%
metrics_names = ['Accuracy', 'Adjusted Accuracy', 'Macro F1-Score', 'Invalid Rate']
bert_values = [bert_acc_sub, bert_acc_sub, bert_f1_sub, 0.0]
llm_values = [overall_accuracy, adjusted_accuracy, macro_f1_llm, invalid_rate]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, bert_values, width, label='ConvBERT', edgecolor='black')
bars2 = ax.bar(x + width/2, llm_values, width, label='Gemini 3.1 Flash Lite', edgecolor='black')

ax.set_ylabel('Score')
ax.set_title(f'ConvBERT vs Gemini 3.1 Flash Lite — Performance Comparison ({total_samples} Test Subset)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# [PLACEHOLDER: Interpretation of the performance comparison bar chart will be added here after the cell output is generated.]

# %% [markdown]
# ### Task 4: Your Work Ends Here
# ---

# %% [markdown]
# ## Task 5: Comparative Report and Discussion
# 
# Write a concise but analytical discussion comparing the fine-tuned BERT model and the LLM-based approach. Your discussion should address the following aspects:
# 
# ### 5.1 Performance vs. Reliability
# 
# * Which model achieved higher accuracy?
# * Did the LLM suffer from formatting inconsistencies?
# * Was the invalid prediction rate significant?
# * Did BERT’s deterministic behavior provide advantages for production usage?
# * Discuss the trade-off between flexibility and reliability.
# 
# ### 5.2 Error Analysis
# 
# Using the confusion matrices:
# 
# * Identify which labels were most difficult for each model.
# * Analyze whether these 2 models struggled particularly with Neutral, Contradiction, or Entailment.
# * Discuss possible reasons behind these behaviors.
# 
# ### 5.3 Cost and Efficiency Analysis
# 
# Compare the approaches in terms of:
# 
# * Hardware requirements
# * Inference latency
# * API costs
# * Memory consumption
# * Ease of deployment
# * Scalability
# * Practical trade-offs between fine-tuned task-specific models and general-purpose LLMs.
# 
# Consider the following scenario:
# 
# > A production RAG (Retrieval-Augmented Generation) system needs to verify whether a generated answer is logically supported by retrieved documents.
# 
# Which approach would you deploy?
# 
# * Fine-tuned BERT classifier?
# * Prompted LLM?
# * Hybrid pipeline?
# 
# Justify your decision in terms of Reliability, Cost-efficiency, Latency, Robustness, and Maintainability.

# %% [markdown]
# ---
# ### Task 5: Your Work Starts Here

# %% [markdown]
# ## Task 5: Comparative Report and Discussion
# 
# > **This section is intentionally left as a placeholder.** The comparative report will be written after all outputs, visualizations, and metrics from Tasks 1–4 have been generated and reviewed.
# 
# ### 5.1 Performance vs. Reliability
# [PLACEHOLDER: Analysis comparing accuracy, invalid rates, and deterministic vs. non-deterministic behavior will be added here after all outputs are available.]
# 
# ### 5.2 Error Analysis
# [PLACEHOLDER: Confusion matrix analysis identifying the most difficult labels for each model will be added here after all outputs are available.]
# 
# ### 5.3 Cost and Efficiency Analysis
# [PLACEHOLDER: Comparison of hardware requirements, inference latency, API costs, memory consumption, ease of deployment, scalability, and the RAG deployment scenario discussion will be added here after all outputs are available.]

# %% [markdown]
# ### Task 3: Your Work Ends Here
# ---

# %% [markdown]
# ## Additional Notes
# 
# * There is no strict format for outputs or results. You are encouraged to do your best.
# * Organize your notebook clearly using:
#   * **Markdown cells** for explanations, observations, and comments
#   * **Code cells** for implementation and experiments
# * Explain important implementation decisions.
# * Include sufficient comments throughout your code.
# * Do **not** include API keys or credentials in your submission.
# * Ensure your experiments are reproducible.
# 
# > **Important:**
# > The sections between two horizontal lines (e.g., “Task 1: Your Work Starts/Ends Here”) are your work areas. Within these sections, you can add as many new cells as needed between the lines to complete your solution. Please ensure that you follow the existing notebook structure and do not delete or modify any pre-existing content outside these marked regions.
# 
# ## Submission Guidelines
# 
# * You must submit **only the notebook file** compressed as a `.zip` archive. Model weights or checkpoints are not required.
# * The notebook filename must follow the format: `<studentID>.ipynb` (e.g., `20212223.ipynb`).
# * The **ZIP file name must also be the same as your student ID** (e.g., `20212223.zip`).
# * Before submission, make sure that:
#     * All notebook cells have been executed
#     * Outputs, plots, and visualizations are visible inside the notebook
#     * The notebook is saved properly with outputs included
# 
# 
# 
# > **Important:**
# > This notebook serves as your final report file. Therefore, submissions with missing outputs, missing visualizations, or improperly named files may receive a score reduction.


