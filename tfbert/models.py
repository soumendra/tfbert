import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Dropout, Dense  # type: ignore
from transformers import TFAutoModel  # type: ignore
tf.get_logger().setLevel("INFO")


class BaseModel(tf.keras.Model):
    def __init__(self, model_name):
        """
        Arguments
        ---------
        model_name: str
        Possible values are - 
            'bert-base-uncased', 
            'distilbert-base-uncased',
        """
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.base_model = TFAutoModel.from_pretrained(self.model_name)
        self.dropout = Dropout(0.3)
        self.dense = Dense(2)

    @tf.function
    def call(self, inputs, training=False):
        # last_hidden_state -> (batch_size, sequence_length, hidden_size)
        last_hidden_state = self.base_model(inputs)[0]
        # cls_embeddings -> (batch_size, hidden_size)
        cls_embeddings = last_hidden_state[:, 0, :]
        output_values = self.dropout(cls_embeddings, training=training)
        logits = self.dense(output_values)
        return logits

