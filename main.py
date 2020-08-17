from tfbert.cola_data import ColaData  # type: ignore
from tfbert.models import BuildModel  # type: ignore
from tfbert.loss import bce  # type: ignore
from config import configs_local

# from transformers import AdamWeightDecay
from tensorflow.keras.optimizers import Adam

exp = ColaData(configs_local.datapath)
exp.train(
    configs_local, BuildModel(configs_local), bce, Adam(learning_rate=configs_local.lr),
)
exp.create_submission()
