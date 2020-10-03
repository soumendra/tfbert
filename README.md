# tfbert
Working with - https://www.kaggle.com/aditya08/tf2-2-fine-tuning-bert-for-seq-classification/

FineTuning Tensorflow BERT for [CoLA In-Domain Open Evaluation](https://www.kaggle.com/c/cola-in-domain-open-evaluation)

## Getting Started
All the experiments are run on `python 3.8.0`.

1. Clone the repository
2. If you do not have python3.8 installed. Run the below steps for easy installation using [asdf](https://asdf-vm.com/). *asdf* allows us to manage multiple runtime versions such for different languages such as `nvm`, `rbenv`, `pyenv`, etc using a CLI tool
	* Install asdf using this [guide](https://asdf-vm.com/#/core-manage-asdf-vm?id=install)
	* Now install `python3.8.0`
	```bash
	asdf plugin add python
	asdf install python 3.8.0
	asdf local python 3.8.0	# sets python3.8 as interpreter for the project
	```
	* Check the set python version
	```bash
	asdf current python
	```
3. Install poetry. [Poetry](https://python-poetry.org/docs/) is a python dependency management & packaging tool. Allows us to declare project libraries dependency & manage them
	```bash
	asdf plugin add poetry
	asdf install poetry latest # current 1.0.10; might need sudo
	asdf local poetry 1.0.10
	```
4. Install all dependencies
	```bash
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
