# Named Entity Recognition (NER) Project

## Overview

This project implements Named Entity Recognition systems for both English and Telugu languages, classifying text entities into four categories: Person, Location, Organization, and Miscellaneous. The repository contains multiple NER approaches ranging from traditional rule-based methods to state-of-the-art deep learning techniques.

## Implemented Approaches

The project explores four distinct NER methodologies with implementations for both English and Telugu:

1. **Rule-based NER**: Leverages linguistic patterns and hand-crafted rules
2. **Conditional Random Fields (CRF)**: Statistical sequence modeling with feature engineering
3. **BiLSTM-CRF**: Neural sequence labeling combining bidirectional LSTMs with CRFs
4. **BERT-based NER**: Fine-tuning of transformer models for contextualized token classification

## Project Structure

```
NER_Project/
├── Rule_based_NER_English.ipynb      # Rule-based approach for English
├── RULE_BASED_NER_TELUGU.ipynb       # Rule-based approach for Telugu
├── CRF_NER_english.ipynb             # CRF implementation for English
├── CRF_NER_telugu.ipynb              # CRF implementation for Telugu
├── BILSTM_CRF_English.ipynb          # BiLSTM-CRF architecture for English
├── BILSTM_CRF_TELUGU.ipynb           # BiLSTM-CRF architecture for Telugu
├── BERT_ENGLISH_NER.ipynb            # BERT-based NER for English
├── BERT_TELUGU.ipynb                 # BERT-based NER for Telugu
├── DATA_ANALYSIS.ipynb               # Analysis of the English dataset
└── README.md                         # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training neural models)
- Google Colab account (all notebooks are designed for Colab environment)

### Installation and Setup

1. Upload the notebook files to Google Colab
2. Select the desired runtime (GPU recommended)
3. Run the cells sequentially - each notebook contains all necessary installation commands

### Dataset Information

The project uses publicly available datasets loaded directly through Hugging Face's `datasets` library:

- **English NER**: CoNLL-2003 dataset
- **Telugu NER**: WikiANN (Panx) dataset

**Important Note**: When loading datasets for the first time, you will be prompted with a message asking:
```
Do you want to continue? [y/N]
```
**You must enter `y` to proceed with the dataset download.**

Example of dataset loading (included in notebooks):
```python
# For English
from datasets import load_dataset
english_dataset = load_dataset("conll2003")  # Will prompt for confirmation

# For Telugu
telugu_dataset = load_dataset("wikiann", "te")  # Will prompt for confirmation
```

## Detailed Methodology

### 1. Rule-based Approach
- **Implementation**: Uses regular expressions, gazetteer lists, and syntactic patterns
- **Advantages**: Interpretable, no training data required
- **Limitations**: Limited coverage, requires linguistic expertise

### 2. CRF-based Approach
- **Implementation**: Sequence labeling with manually engineered features
- **Features used**: Word identity, POS tags, capitalization, surrounding context
- **Training**: Maximum likelihood estimation with L2 regularization
- **Inference**: Viterbi algorithm for optimal tag sequence

### 3. BiLSTM-CRF Approach
- **Architecture**: Word embeddings → BiLSTM layers → CRF layer
- **Word representations**: Pre-trained GloVe/FastText embeddings with character-level CNNs
- **Hyperparameters**: Hidden dimensions: 256, Dropout: 0.5, Optimizer: Adam
- **Training strategy**: Early stopping based on validation F1 score

### 4. BERT-based Approach
- **Models used**: 
  - English: bert-base-cased
  - Telugu: indic-transformers/ai4bharat-indic-bert
- **Fine-tuning**: Token classification head on top of transformer encodings
- **Training parameters**: Learning rate: 3e-5, Batch size: 16, Epochs: 3-5
- **Implementation**: Hugging Face's Transformers library

## Performance Evaluation

Each notebook includes comprehensive evaluation sections measuring:
- Precision, recall, and F1 scores per entity type
- Confusion matrices visualizing error patterns
- Cross-validation results where applicable
- Comparative analysis between different approaches

## Requirements

Core dependencies (automatically installed in notebooks):
```
tensorflow>=2.4.0
torch>=1.8.0
transformers>=4.5.0
sklearn-crfsuite>=0.3.6
nltk>=3.6.2
datasets>=1.8.0
pandas>=1.2.4
numpy>=1.19.5
matplotlib>=3.4.2
seaborn>=0.11.1
```

## Troubleshooting Guide

### Common Issues and Solutions

- **Memory errors in Colab**: 
  - Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
  - Reduce batch sizes in model training sections
  - Restart runtime after installing memory-intensive packages

- **Dataset loading problems**:
  - Ensure you type `y` when prompted for dataset download confirmation
  - Check internet connection stability
  - Try restarting the runtime if downloads timeout

- **Training takes too long**:
  - Reduce the number of epochs
  - Use smaller subsets of data for initial experiments
  - Enable GPU acceleration

- **Out of memory during BERT training**:
  - Reduce batch size
  - Use gradient accumulation
  - Consider upgrading to Colab Pro for more memory

## Contact Information

For questions or issues, please contact:

**Shriyatha**
- Email: 142201033@smail.iitpkd.ac.in
- GitHub: [https://github.com/Shriyatha](https://github.com/Shriyatha)

**Jayati**
- Email: 142201006@smail.iitpkd.ac.in

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing the datasets API and transformers library
- The creators and contributors of the CoNLL-2003 and WikiANN datasets
- The research community for developing various NER techniques

## Links to Resources

- [CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)
- [WikiANN Dataset](https://huggingface.co/datasets/wikiann)
- [BERT Original Paper](https://arxiv.org/abs/1810.04805)
- [BiLSTM-CRF for Sequence Tagging](https://arxiv.org/abs/1508.01991)
- [CRF Implementation Documentation](https://sklearn-crfsuite.readthedocs.io/)
