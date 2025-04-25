# Named Entity Recognition PROJECT

Named Entity Recognition for Both English And Telugu Datasets - Classifying them into entities namely, Person, Location, Organisation, Miscellaneous.

This repository contains implementations of Named Entity Recognition (NER) systems for both English and Telugu languages using multiple approaches:

1. Rule-based NER
2. Conditional Random Fields (CRF) based NER
3. BiLSTM-CRF based NER 
4. BERT-based NER

## Project Structure

This project consists of 7 Jupyter Notebook (.ipynb) files that implement different NER techniques for both English and Telugu languages:

1. `Rule_based_NER_English.ipynb` - Rule-based approach for English NER
2. `RULE_BASED_NER_TELUGU.ipynb` - Rule-based approach for Telugu NER
3. `CRF_NER_english.ipynb` - CRF-based NER implementation English NER
4. `CRF_NER_telugu` - CRF-based NER implementation Telugu NER
5. `BILSTM_CRF_English.ipynb` - BiLSTM-CRF architecture for English
6. `BILSTM_CRF_TELUGU.ipynb` - BiLSTM-CRF architecture for Telugu
7. `BERT_ENGLISH_NER.ipynb` - BERT-based NER for English
8. `BERT_TELUGU.ipynb` - BERT-based NER for Telugu

## Getting Started

## Configuration and Installation

Prerequisites

Python 3.8+
CUDA-compatible GPU (recommended for neural network training)


### Running the Notebooks

All notebooks are designed to be run in Google Colab:

1. Upload the notebook files to Google Colab
2. Open each notebook in Colab by clicking on it
3. Run the cells in sequence

The notebooks include all necessary installation commands and will automatically download required libraries when executed.

### Dataset Information

This project uses the following publicly available datasets that are imported directly in the notebooks:

- **English NER**: CoNLL-2003 dataset via Hugging Face's `datasets` library
- **Telugu NER**: WikiANN (Panx) dataset via Hugging Face's `datasets` library

Example of dataset loading (included in the notebooks):
```python
# For English
from datasets import load_dataset
english_dataset = load_dataset("conll2003")

# For Telugu
telugu_dataset = load_dataset("wikiann", "te")
```

File Manifest

NER_Project/
├── Rule_based_NER_English.ipynb      # Rule-based approach for English NER
├── RULE_BASED_NER_TELUGU.ipynb       # Rule-based approach for Telugu NER
├── CRF_NER_english.ipynb             # CRF-based NER implementation for English
├── CRF_NER_telugu.ipynb              # CRF-based NER implementation for Telugu
├── BILSTM_CRF_English.ipynb          # BiLSTM-CRF architecture for English NER
├── BILSTM_CRF_TELUGU.ipynb           # BiLSTM-CRF architecture for Telugu NER
├── BERT_ENGLISH_NER.ipynb            # BERT-based NER for English
├── BERT_TELUGU.ipynb                 # BERT-based NER for Telugu
├── DATA_ANALYSIS.ipynb               # Anaylsing the English Dataset
├── README.md                     



## Methodology

### Rule-based Approach
Implements hand-crafted rules and patterns to identify named entities in text using linguistic features and regular expressions.

### CRF-based Approach
Uses Conditional Random Fields with manually engineered features to capture context and make sequence predictions.

### BiLSTM-CRF Approach
Combines bidirectional LSTM networks with CRF to leverage both deep learning capabilities and structured prediction.

### BERT-based Approach
Fine-tunes pre-trained BERT models for the NER task, utilizing contextualized word embeddings.

## Requirements

All requirements are installed automatically within the notebooks. The main dependencies include:

- tensorflow
- torch
- transformers
- sklearn-crfsuite
- nltk
- datasets
- pandas
- numpy
- matplotlib
- seaborn

## Performance

Each notebook includes evaluation sections that measure:
- Precision, Recall, and F1 scores per entity type
- Overall accuracy metrics
- Confusion matrices for error analysis

## Known Issues

- The rule-based approach for Telugu may require additional refinement for certain entity types
- Large BERT models may experience timeout issues in the free version of Colab due to memory constraints

## Troubleshooting

- **Memory issues in Colab**: Consider enabling GPU in Colab: Runtime > Change runtime type > Hardware accelerator > GPU
- **Dataset loading errors**: Check your internet connection, as the datasets are downloaded on-the-fly
- **Out of memory errors**: Try reducing batch sizes in the respective notebooks

## Contact Information

For questions or issues, please contact:
- [Shriyatha]
- [142201033@smail.iitpkd.ac.in]
- [https://github.com/Shriyatha]

- [Jayati]
- [142201006@smail.iitpkd.ac.in]


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Hugging Face for providing the datasets API
- The CoNLL-2003 and WikiANN dataset creators and contributors
- The research community for developing the different NER techniques implemented here

## Links to Relevant Resources

- [CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)
- [WikiANN Dataset](https://huggingface.co/datasets/wikiann)
- [BERT Model](https://github.com/google-research/bert)
- [CRF Implementation](https://sklearn-crfsuite.readthedocs.io/)
