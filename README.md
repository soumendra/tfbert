# tfbert
Go away


# Working with it

https://www.kaggle.com/aditya08/tf2-2-fine-tuning-bert-for-seq-classification/

It uses poetry. After cloning,

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

To train the model,

```bash
poetry run python main.py
```

Currently stuck at `current_error.png`

# Later

https://www.kaggle.com/aditya08/tf2-2-k-fold-bert-fine-tuning
