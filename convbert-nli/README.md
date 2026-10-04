# Natural Language Inference with ConvBERT

An end-to-end Natural Language Inference (NLI) study on the Stanford Natural Language Inference (SNLI) dataset. The project fine-tunes ConvBERT to classify premise–hypothesis pairs as **entailment**, **neutral**, or **contradiction**, then compares it with a prompted large language model on the same test subset.

## Pipeline

1. Load and validate the official SNLI train, validation, and test splits.
2. Remove invalid labels and duplicate sentence pairs.
3. Analyze label balance, sentence lengths, vocabulary, OOV behavior, lexical overlap, and UMAP projections.
4. Expand English contractions while preserving punctuation and stopwords.
5. Tokenize sentence pairs as `[CLS] premise [SEP] hypothesis [SEP]` with segment IDs and attention masks.
6. Fine-tune the last five ConvBERT encoder layers plus the classifier.
7. Select the best checkpoint by validation accuracy and evaluate once on the full test split.
8. Compare ConvBERT with Gemini 3.1 Flash Lite on the same 1,000 test samples.

## Model and Training Setup

- Base model: `YituTech/conv-bert-base`
- Dataset: `stanfordnlp/snli`
- Train subset: 100,000 stratified examples
- Batch size: 64
- Learning rate: 2e-5
- Epochs: 5
- Optimizer: AdamW
- Scheduling: linear warmup/decay
- Padding: dynamic, per batch
- Selection metric: validation accuracy

## Results

| Evaluation | Accuracy | Macro F1 |
| --- | ---: | ---: |
| ConvBERT — full SNLI test set (9,824 samples) | 88.57% | 88.54% |
| ConvBERT — shared 1,000-sample subset | 87.70% | 87.67% |
| Gemini 3.1 Flash Lite — same subset | 92.40% | 92.39% |

The full ConvBERT test classification report shows the strongest recall on contradiction (91%) and the main remaining difficulty on neutral examples (84%).

## Files

- `convbert_nli.ipynb`: complete experiment with saved plots, logs, and evaluation outputs
- `convbert_nli.py`: script export of the notebook
- `requirements.txt`: Python dependencies

The trained `best_convbert_nli.pt` checkpoint is not stored in Git because it is over 400 MB. Running the training section recreates it locally.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook convbert_nli.ipynb
```

Training benefits substantially from a CUDA-capable GPU. The optional LLM comparison requires a Google AI Studio API key; keep the key outside source control and replace the placeholder only in your private runtime environment.
