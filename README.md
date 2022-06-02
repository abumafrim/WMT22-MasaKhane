## Fintuning [ALBERT](https://openreview.net/forum?id=H1eA7AEtvS) for Sentence-pair Classification

This repository contains the code for training a sentence-pair classification model. The codes were adapted from [Fine-tune ALBERT for sentence-pair classification](https://colab.research.google.com/github/NadirEM/nlp-notebooks/blob/master/Fine_tune_ALBERT_sentence_pair_classification.ipynb#scrollTo=SZIHhCYNhN-H). The training data consist of positive sentences (correct translations) from the [MAFAND](https://github.com/masakhane-io/lafand-mt/tree/main/data/text_files) dataset and a sample of negative samples (wrong translations), based on [laser scores](https://github.com/facebookresearch/LASER), from the [wmt22_african](https://github.com/facebookresearch/LASER/tree/main/data/wmt22_african) dataset.

For the license of the data, see [MAFAND](https://github.com/masakhane-io/lafand-mt/tree/main/data/text_files) and [wmt22_african](https://github.com/facebookresearch/LASER/tree/main/data/wmt22_african).

### Setting up
```bash
pip install -r requirements.txt
```

### Fine-tuning
To run the code, see any of the bash scripts (*.sh)

#### For 'albert-base-v2'
```bash
bash albert.sh
```

#### For 'bert-base-uncased'
```bash
bash bert-base-uncased.sh
```

### Prediction
To use the fine-tuned model to predict correct translations, supply correct parameters in predict.sh and run
```bash
bash predict.sh
```