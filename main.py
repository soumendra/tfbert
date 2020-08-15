from tfbert.cola_data import ColaData  # type: ignore
from tfbert.models import BertModel  # type: ignore
from tfbert.loss import bce  # type: ignore
from config import configs_local
from transformers import AdamWeightDecay, WarmUp

exp = ColaData(configs_local.datapath)
exp.train(configs_local, BertModel(), bce, AdamWeightDecay(learning_rate=configs_local.lr))
exp.create_submission()
