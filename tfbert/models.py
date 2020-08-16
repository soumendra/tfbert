import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Dropout, Dense  # type: ignore
from transformers import TFBertModel  # type: ignore
tf.get_logger().setLevel("INFO")


class BertModel(tf.keras.Model):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.dropout = Dropout(0.3)
        self.dense = Dense(2)

    @tf.function
    def call(self, inputs, training=False):
        """
        Last layer hidden-state of [CLS] token further processed by a 
        Linear layer and a Tanh activation function - (batch_size, 768) 
        """
        _, pooled_output = self.bert(inputs)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.dense(pooled_output)
        return logits
