# tfbert
Working with - https://www.kaggle.com/aditya08/tf2-2-fine-tuning-bert-for-seq-classification/

FineTuning Tensorflow BERT for [CoLA In-Domain Open Evaluation](https://www.kaggle.com/c/cola-in-domain-open-evaluation)

## Setup
It uses poetry. After cloning - 

```bash
poetry install
```

If you do not have `python 3.8` as the default python or a virtual environment based on python 3.8, you might run into some python version issues. In such cases, follow the below steps -
```bash
# install virtualenv and pyenv
brew install pyenv pyenv-virtualenv
# install python3.8.0
pyenv install 3.8.0
# create a virtualenv with python3.8.0
pyenv virtualenv 3.8.0 dl_env
# activate the environment
source ~/.pyenv/versions/dl_env/bin/activate
# install dependencies
poetry install
```

## Running
To train a model & generate *submission file* - 

```bash
poetry run python main.py
```
To choose between multiple models, change *model_name* in `config` to one of the following -
* `bert-base-uncased`
* `distilbert-base-uncased`
* `roberta-base`

## Structure
```bash
├── config.py
├── data
│   ├── cola_in_domain_test.tsv
│   └── cola_public_1.1
│       └── cola_public
│           ├── raw
│           └── tokenized
├── main.py
├── README.md
└── tfbert
    ├── cola_data.py    -> ColaData Class - training BERT & creating submission file
    ├── dataset.py      -> BertDataset Class - returns tf.data.Dataset object 
    ├── loss.py         -> Loss function
    ├── metrics.py      -> Custom metrics
    └── models.py       -> BaseModel Class - returns a TF Model of model_name in config 
```

## General
* Managing Python version: `asdf`
* Python dependancy/project management: `poetry`
* Data/model versioning/management: `dvc`
* Auto-linting: `black`

# Future

https://www.kaggle.com/aditya08/tf2-2-k-fold-bert-fine-tuning
