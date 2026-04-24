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
- Aggregated from multiple medical QA sub-corpora with domain-specific medical vocabulary
- Manually split 80/10/10 into train, validation, and test sets (no pre-built splits available)
- Minimal missing data; see `EDA_Medical_QA.ipynb` for full dataset analysis

**License:** Please refer to the original dataset page for licensing and citation requirements before use.

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
## Quick Start

### Requirements

- Python 3.10+
- GPU strongly recommended for transformer fine-tuning (training on CPU will be very slow)

### Installation

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### Run

1. **Explore the data** — open and run `EDA_Medical_QA.ipynb` end-to-end
2. **Train and evaluate models** — open and run `Model_Medical_QA.ipynb` end-to-end

> **Expected runtime:** EDA notebook runs in ~5 minutes. Model training takes approximately 1–3 hours per model on a GPU (longer on CPU). Inference and evaluation on the 500-example test sample takes ~10–20 minutes.

---

## Usage Guide

### Step 1: EDA (`EDA_Medical_QA.ipynb`)
- Loads the HuggingFace dataset automatically
- Generates summary statistics, length distributions, and vocabulary analysis
- Outputs plots saved to `results/figures/`

### Step 2: Model Training & Evaluation (`Model_Medical_QA.ipynb`)
- Trains the LSTM baseline from scratch
- Fine-tunes DistilGPT-2 using HuggingFace `Trainer`
- Evaluates both models on a 500-example test sample
- Generates and saves ROUGE score plots and training curves to `results/figures/`
- Saves the best LSTM checkpoint to `models/lstm_best.pt`

To run inference on a trained model without retraining, skip the training cells and load the saved checkpoint:
```python
# LSTM
model.load_state_dict(torch.load("models/lstm_best.pt"))

# DistilGPT-2
model = AutoModelForCausalLM.from_pretrained("./distilgpt2-finetuned")
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


Key packages:
- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `rouge_score`
- `nltk`
- `matplotlib`
- `seaborn`
- `tqdm`

See `requirements.txt` for exact versions.

---

## Authors and Contributions

| Name | Role |
|---|---|
| **Lauren Plummer** | EDA notebook, model training & evaluation, README |
| **Megan LeComte** | Figures, EDA support |

