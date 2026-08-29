# Nepali News Classifier

A text classification system that categorizes Nepali-language news articles into six topic categories using a fine-tuned BERT model.

## Categories

| Category  | Description                    |
|-----------|--------------------------------|
| economy   | Financial and economic news    |
| global    | International affairs          |
| health    | Health and medical news        |
| national  | Domestic general news          |
| politics  | Political news and analysis    |
| sports    | Sports coverage                |

## Project Structure

```
.
├── prepare_data.py      # Download, filter, clean, split dataset
├── train.py             # Fine-tune NepaliBERT, evaluate on test set
├── classify_cli.py      # Interactive CLI for real-time classification
├── stats.ipynb          # Training statistics and visualizations
├── requirements.txt     # Python dependencies
├── data/                # Generated CSVs and label mappings
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── labels.json
└── model/               # Fine-tuned model and checkpoints
```

## Setup

### Option A: venv

```bash
python -m venv .ai_project
source .ai_project/bin/activate
pip install -r requirements.txt
pip install matplotlib jupyter sentencepiece
```

### Option B: conda

```bash
conda create -n nepali-news python=3.11 -y
conda activate nepali-news
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install matplotlib jupyter sentencepiece
```

For CPU-only conda:

```bash
conda create -n nepali-news python=3.11 -y
conda activate nepali-news
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
pip install matplotlib jupyter sentencepiece
```

## Usage

### 1. Prepare Data

Downloads the `spandyie/nepali-news-dataset` from HuggingFace Hub, filters to the six target categories, and produces balanced train/val/test splits.

```bash
python prepare_data.py
```

Outputs 90,000 samples (15,000 per category) split 80/10/10.

### 2. Train Model

Fine-tunes `Rajan/NepaliBERT` on the prepared data.

```bash
python train.py
```

Training configuration:

| Parameter            | Value              | Purpose                                      |
|----------------------|--------------------|----------------------------------------------|
| Base model           | Rajan/NepaliBERT   | Pretrained BERT for Nepali language          |
| Max sequence length  | 256                | Captures ~3-4 sentences of text              |
| Batch size           | 32                 | Samples per device per step                  |
| Gradient accumulation| 2                  | Effective batch size = 64                    |
| Epochs               | 15                 | Maximum; early stopping may halt earlier     |
| Learning rate        | 1e-5               | Peak LR after warmup                         |
| Warmup steps         | 900                | Linear ramp from 0 to 1e-5                   |
| Weight decay         | 0.01               | L2 regularization                            |
| Early stopping       | patience=3         | Stops if val accuracy plateaus for 3 epochs  |
| Mixed precision      | fp16 on CUDA       | Faster training, lower VRAM usage            |

#### Learning rate schedule

The training uses a linear warmup + linear decay schedule:

1. **Steps 0-900 (warmup):** Learning rate increases linearly from 0 to 1e-5. This prevents large, destabilizing updates at the start when the classification head is randomly initialized.
2. **Steps 900-end (decay):** Learning rate decreases linearly from 1e-5 to ~0. This lets the model settle into a sharp minimum without oscillating.

### 3. Classify Text

Interactive CLI that accepts Nepali text input and returns the predicted category with confidence scores.

```bash
python classify_cli.py
```

Example output:

```
Enter text: काठमाडौंमा आज संसदको बैठक भयो

  Predicted category: politics (confidence: 87.34%)
  Top-3:
    1. politics: 87.34%
    2. national: 8.21%
    3. global: 2.15%
```

### 4. View Statistics

Run the Jupyter notebook for training curves, confusion matrix, per-class metrics, and confidence analysis.

```bash
jupyter notebook stats.ipynb
```

## Model Performance

Results on the held-out test set (9,000 samples):

| Category  | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| economy   | 0.85      | 0.83   | 0.84     |
| global    | 0.87      | 0.87   | 0.87     |
| health    | 0.79      | 0.73   | 0.76     |
| national  | 0.68      | 0.77   | 0.72     |
| politics  | 0.88      | 0.82   | 0.84     |
| sports    | 0.96      | 0.99   | 0.98     |

**Overall accuracy: 83.1%**

Sports classification achieves near-perfect results. Economy, global, and politics are reliably identified. National and health show moderate confusion due to overlapping subject matter (e.g., government health policy classified as national news).

## Data Source

Dataset: [spandyie/nepali-news-dataset](https://huggingface.co/datasets/spandyie/nepali-news-dataset) on HuggingFace Hub.

Articles are streamed, cleaned (HTML/URL removal, sentence truncation to 3 sentences), deduplicated, and balanced across categories.

## Dependencies

- Python 3.10+
- transformers
- torch
- datasets
- scikit-learn
- pandas
- accelerate
- matplotlib (for notebook)
- sentencepiece (for tokenizer)
