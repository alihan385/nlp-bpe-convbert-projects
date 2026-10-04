# Byte Pair Encoding Tokenizer from Scratch

This project implements Byte Pair Encoding (BPE) without a tokenization library. It learns frequent adjacent-symbol merges from a training corpus, grows a subword vocabulary, and applies the learned rules to unseen text.

## What It Implements

- Character-level vocabulary initialization with Turkish characters, letters, digits, and punctuation
- Adjacent-pair frequency counting
- Deterministic pair ranking by frequency, merged-token length, and lexical order
- Configurable `max_merge_count` and `topK` merge selection
- Tokenization of test corpora with the learned merge table
- Conversion of subword tokens into vocabulary IDs

## Core API

```python
merges, vocabulary = train_BPE(
    "data/train.txt",
    init_vocab,
    max_merge_count=20,
    topK=3,
)

tokens, input_ids = test_BPE(
    "data/test.txt",
    merges,
    vocabulary,
)
```

## Run

Requires Python 3.9+ and no third-party packages.

```bash
python bpe_tokenizer.py
```

The script runs eight configurations against the included corpora and writes `output.txt`, `output1.txt`, and `output2.txt` in this directory.

## Files

- `bpe_tokenizer.py`: complete implementation and experiment runner
- `data/train*.txt`: training corpora
- `data/test*.txt`: evaluation corpora

## Design Notes

Words are joined with explicit underscore boundaries before merge learning. This lets the algorithm retain boundary information while repeatedly replacing the most useful adjacent pairs. The learned merge sequence is replayed in the same order during inference, ensuring deterministic tokenization.
