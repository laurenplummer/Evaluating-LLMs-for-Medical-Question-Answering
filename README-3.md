# Evaluating LLMs for Medical Question Answering

A comparison of LSTM and Transformer (DistilGPT-2) language models for medical question answering, evaluated using perplexity and ROUGE scores.

---

## Overview

This project trains and evaluates two language model architectures on a large medical QA dataset to assess their ability to generate clinically relevant responses to patient questions. The models are compared on loss, perplexity, and ROUGE metrics.

---

## Dataset

**[Malikeh1375/medical-question-answering-datasets](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets)** via HuggingFace Datasets

- ~247,000 medical QA examples
- Each example contains a patient question (`instruction` + `input`) and a doctor's response (`output`)
- Aggregated from multiple medical QA sub-corpora
- Manually split 80/10/10 into train, validation, and test sets (no pre-built splits)

---

## Project Structure

```
├── EDA_Medical_QA.ipynb          # Exploratory data analysis
├── Model_Medical_QA.ipynb        # Model training & evaluation
├── results/
│   └── figures/                  # Training curves, ROUGE plots
├── models/
│   └── lstm_best.pt              # Best LSTM checkpoint (saved locally)
└── README.md
```

---

## Models

### LSTM Baseline
- Custom LSTM language model trained from scratch
- Vocabulary built from training corpus
- Greedy decoding for text generation
- Trained for 3 epochs with Adam optimizer (lr=3e-3), gradient clipping, and ReduceLROnPlateau scheduler

### Transformer (DistilGPT-2)
- Fine-tuned [`distilgpt2`](https://huggingface.co/distilgpt2) from HuggingFace
- Fine-tuned for 3 epochs using HuggingFace `Trainer`
- Cosine LR schedule (lr=5e-5), mixed precision (fp16), dynamic padding
- Greedy decoding with `model.generate()`

---

## Evaluation

Both models are evaluated on a 500-example sample of the test set using:

| Metric | Description |
|---|---|
| **Cross-Entropy Loss** | Average token-level loss |
| **Perplexity** | `exp(loss)` — lower is better |
| **ROUGE-1/2/L** | N-gram overlap between generated and reference responses |

---

## Setup & Usage

### Install dependencies
```bash
pip install torch transformers datasets evaluate rouge_score nltk matplotlib seaborn tqdm
```

### Run notebooks in order
1. **`EDA_Medical_QA.ipynb`** — explore and understand the dataset
2. **`Model_Medical_QA.ipynb`** — train models and evaluate

---

## Key EDA Findings

- Combined input (instruction + context) has a median length of ~50 words; outputs average longer
- Dataset aggregates several medical sub-corpora with varying size and style
- Minimal missing data; vocabulary is domain-specific with heavy medical terminology

---

## Requirements

- Python 3.10+
- PyTorch
- HuggingFace `transformers`, `datasets`, `evaluate`
- `rouge_score`, `nltk`, `matplotlib`, `seaborn`, `tqdm`

GPU recommended for transformer fine-tuning.

---

## Author

Lauren Plummer
