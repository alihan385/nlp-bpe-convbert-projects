# NLP Projects: BPE Tokenization and ConvBERT NLI

Two natural language processing projects covering both classical subword tokenization and modern transformer fine-tuning.

## Projects

| Project | Focus | Highlights |
| --- | --- | --- |
| [BPE Tokenizer](./bpe-tokenizer/) | Byte Pair Encoding from scratch | Corpus-driven merge learning, configurable top-k selection, vocabulary construction, token IDs |
| [ConvBERT NLI](./convbert-nli/) | Natural Language Inference on SNLI | Data analysis, preprocessing, partial fine-tuning, full test evaluation, LLM comparison |

## Key Results

The ConvBERT model reached **88.57% accuracy** and **88.54% macro F1** on all 9,824 valid SNLI test examples. On the shared 1,000-example comparison subset, ConvBERT reached **87.70% accuracy**, while Gemini 3.1 Flash Lite reached **92.40% accuracy**.

## Repository Structure

```text
.
├── bpe-tokenizer/
│   ├── data/
│   ├── bpe_tokenizer.py
│   └── README.md
├── convbert-nli/
│   ├── convbert_nli.ipynb
│   ├── convbert_nli.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

Large generated artifacts, duplicate submission archives, and the 400+ MB model checkpoint are intentionally excluded. The notebook retains the experiment outputs and reported metrics.

## Author

Alihan Ertekin
