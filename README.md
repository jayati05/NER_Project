# 🔍 Named Entity Recognition (NER) Project

![NER Banner](https://via.placeholder.com/800x200?text=Named+Entity+Recognition+Project)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)

## 📋 Overview

A comprehensive Named Entity Recognition system for English and Telugu languages that identifies and classifies text entities into four categories: **Person**, **Location**, **Organization**, and **Miscellaneous**. This project implements multiple state-of-the-art approaches, from traditional rule-based methods to advanced neural architectures.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Shriyatha/ner-project.git
cd ner-project

# Open notebooks in Google Colab
# OR run locally with:
pip install -r requirements.txt
jupyter notebook
```

## 💻 Implementation Approaches

| Approach | Description | English | Telugu |
|----------|-------------|:-------:|:------:|
| **Rule-based** | Pattern matching using linguistic rules | ✅ | ✅ |
| **CRF** | Statistical modeling with engineered features | ✅ | ✅ |
| **BiLSTM-CRF** | Neural networks with sequential tagging | ✅ | ✅ |
| **BERT** | Transformer-based contextualized embeddings | ✅ | ✅ |

## 📁 File Manifest

```
NER_Project/
├── Rule_based_NER_English.ipynb    # Rule-based approach for English
├── RULE_BASED_NER_TELUGU.ipynb     # Rule-based approach for Telugu
├── CRF_NER_english.ipynb           # CRF implementation for English
├── CRF_NER_telugu.ipynb            # CRF implementation for Telugu
├── BILSTM_CRF_English.ipynb        # BiLSTM-CRF architecture for English
├── BILSTM_CRF_TELUGU.ipynb         # BiLSTM-CRF architecture for Telugu
├── BERT_ENGLISH_NER.ipynb          # BERT-based NER for English
├── BERT_TELUGU.ipynb               # BERT-based NER for Telugu
├── DATA_ANALYSIS.ipynb             # Dataset analysis and visualization
├── requirements.txt                # Project dependencies
└── README.md                       # This documentation file
```

## ⚙️ Configuration & Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab account (preferred environment)

### Setup Instructions
1. Upload notebooks to Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → Hardware accelerator → GPU`
3. Execute cells sequentially - installation commands are included in each notebook

### Dataset Usage
The project automatically loads datasets via Hugging Face:

```python
# When prompted with "Do you want to continue? [y/N]"
# YOU MUST TYPE 'y' TO PROCEED
from datasets import load_dataset
english_dataset = load_dataset("conll2003")     # English dataset
telugu_dataset = load_dataset("wikiann", "te")  # Telugu dataset
```

## 🔧 Technical Implementation

### Rule-based NER
- **Implementation**: Leverages regex patterns, gazetteers, and syntactic rules
- **Performance**: High precision but limited recall
- **Use case**: Domains with predictable entity patterns

### CRF-based NER
- **Features**: Word forms, POS tags, capitalization, n-grams, context windows
- **Training**: L-BFGS with L2 regularization, 5-fold cross-validation
- **Inference**: Viterbi algorithm for optimal sequence decoding

### BiLSTM-CRF Architecture
```
Input → Embedding Layer → BiLSTM → CRF → Entity Tags
```
- **Embeddings**: GloVe (English), FastText (Telugu) with character CNNs
- **Network**: 2-layer BiLSTM (256 units), dropout 0.5
- **Training**: Adam optimizer, early stopping, batch size 32

### BERT-based NER
- **Models**: 
  - English: bert-base-cased
  - Telugu: ai4bharat-indic-bert
- **Fine-tuning**: Linear classification layer on top of token embeddings
- **Training**: Learning rate 3e-5, warmup, weight decay

## 📊 Performance Results

| Model | English F1 | Telugu F1 | Training Time |
|-------|:----------:|:---------:|:-------------:|
| Rule-based | 67.8% | 59.2% | - |
| CRF | 83.5% | 72.6% | ~15 min |
| BiLSTM-CRF | 89.3% | 78.4% | ~1 hour |
| BERT | 91.7% | 81.2% | ~2 hours |

## 📦 Dependencies

```
tensorflow>=2.4.0        # Neural network framework
torch>=1.8.0             # PyTorch framework
transformers>=4.5.0      # BERT models
evaluate                 # Evaluation metrics
seqeval                  # Sequence labeling evaluation
sklearn-crfsuite>=0.3.6  # CRF implementation
nltk>=3.6.2              # NLP utilities
datasets>=1.8.0          # Dataset loading
tqdm                     # Progress bars
torchcrf                 # CRF for PyTorch
pytorch-crf              # Alternative CRF implementation
tabulate                 # Result formatting
spacy                    # NLP pipeline
pandas>=1.2.4            # Data manipulation
numpy>=1.19.5            # Numerical operations
matplotlib>=3.4.2        # Visualization
seaborn>=0.11.1          # Enhanced visualization
```

## ⚠️ Known Issues & Troubleshooting

### Memory Issues
- **Symptom**: Runtime crashes during model training
- **Solution**: Reduce batch size, enable GPU, restart runtime after installations
- **Prevention**: Use progressive model building with checkpoints

### Dataset Loading Problems
- **Symptom**: Dataset download fails or hangs
- **Solution**: Ensure you type `y` when prompted, check connection
- **Alternative**: Download datasets manually using provided scripts

### Telugu Model Performance
- **Issue**: Lower performance on Telugu compared to English
- **Workaround**: Increase training data with augmentation techniques
- **Future Work**: Implement custom tokenizers for Telugu morphology

## 👨‍💻 Contact Information

**Shriyatha** (Project Lead)  
📧 Email: 142201033@smail.iitpkd.ac.in  
🔗 GitHub: [https://github.com/Shriyatha](https://github.com/Shriyatha)

**Jayati** (ML Architecture)  
📧 Email: 142201006@smail.iitpkd.ac.in

## ©️ License & Copyright

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

```
Copyright (c) 2025 Shriyatha, Jayati

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files.
```

## 🙏 Credits & Acknowledgments

- **Hugging Face** for datasets API and transformers library
- **CoNLL-2003** and **WikiANN** dataset creators
- **AI4Bharat** for Indic language models and resources
- **IIT Palakkad** for computational resources and academic guidance

## 📚 Resources & References

- [CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)
- [WikiANN Dataset](https://huggingface.co/datasets/wikiann)
- [BERT Paper: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Neural Architectures for NER](https://arxiv.org/abs/1603.01360)
- [Transfer Learning for NER](https://www.aclweb.org/anthology/N19-1078/)

---

<div align="center">
<img src="https://via.placeholder.com/40" alt="Logo" width="40">
<br>
<i>Advancing NLP for English and Low-resource Indian Languages</i>
</div>
