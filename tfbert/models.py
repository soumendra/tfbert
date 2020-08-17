import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dropout, Dense, Input  # type: ignore
from transformers import TFAutoModel  # type: ignore


def BuildModel(config):
    inputs = Input(shape=(config.max_len), dtype=tf.int32)
    base_model = TFAutoModel.from_pretrained(config.model_name)
    # last_hidden_state -> (batch_size, sequence_length, hidden_size)
    last_hidden_state = base_model(inputs)[0]
    # cls_embeddings -> (batch_size, hidden_size)
    cls_embedd = last_hidden_state[:, 0, :]
    out_values = Dropout(0.3)(cls_embedd)
    out_values = Dense(2, activation="softmax")(out_values)
    model = Model(inputs=inputs, outputs=out_values)
    return model
