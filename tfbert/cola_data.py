import pandas as pd  # type: ignore
from pathlib import Path
from pandas import DataFrame
from typing import List, Union
from tensorflow.keras.utils import to_categorical  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import tensorflow as tf  # type: ignore
from attrdict import AttrDict  # type: ignore
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
from sklearn.metrics import accuracy_score, matthews_corrcoef  # type: ignore

tf.get_logger().setLevel("INFO")


class BertDataset:
    def __init__(self, max_len, model_name, batch_size):
        self.max_len = max_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.batch_size = batch_size

    def encode_text(self, sentences):
        sentences = [" ".join(sentence.split()) for sentence in sentences]
        encoded = self.tokenizer.batch_encode_plus(sentences, max_length=self.max_len, pad_to_max_length=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return (input_ids, attention_mask)

    def create(self, sentences, labels, training=False):
        input_ids, attention_mask = self.encode_text(sentences)
        dataset = tf.data.Dataset.from_tensor_slices((input_ids, labels))
        if training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class ColaData:
    @staticmethod
    def describe_df(df: DataFrame, label: str) -> None:
        print(f"{label}\nShape: {df.shape}\nDistribution:\n{df['label'].value_counts()}\n")

    @staticmethod
    def get_cola_xy(df: DataFrame) -> List[DataFrame]:
        return [df["sentence"], df["label"]]

    def get_cola_df(self):
        in_domain_train = pd.read_csv(self.path / "in_domain_train.tsv", sep="\t", names=self.cols)
        in_domain_val = pd.read_csv(self.path / "in_domain_dev.tsv", sep="\t", names=self.cols)
        out_domain_val = pd.read_csv(self.path / "out_of_domain_dev.tsv", sep="\t", names=self.cols)
        val = in_domain_val.append(out_domain_val)
        test = pd.read_csv(self.path / "../../../cola_out_of_domain_test.tsv", sep="\t")
        return [in_domain_train, val, test]

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.cols = ["source", "label", "notes", "sentence"]
        self.traindf, self.valdf, self.testdf = self.get_cola_df()
        self.x_train, self.y_train = self.get_cola_xy(self.traindf)
        self.x_val, self.y_val = self.get_cola_xy(self.valdf)

        self.y_train_enc = to_categorical(self.y_train)
        self.y_val_enc = to_categorical(self.y_val)

        print(f"\nColaData instantiated from path: {self.path}\n")
        self.describe_df(self.traindf, "Train Data")
        self.describe_df(self.valdf, "Val Data")

    def _train(self, config: AttrDict, model, loss_fn, optimizer):
        @tf.function
        def train_step(model, inputs, y, optimizer):
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = loss_fn(y, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, logits

        losses: List = []
        for idx, (input_ids, label) in tqdm(enumerate(self.train_dataset)):
            loss, logits = train_step(model, input_ids, label, optimizer)
            if idx % 50 == 0 and idx != 0:
                print(f"Batch: {idx}\tLoss: {np.mean(losses)}")
            losses.append(loss)
        return self

    def train(self, config: AttrDict, model, loss_fn, optimizer):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_bert_dataset = BertDataset(config.max_len, config.model_name, config.train_batch_size)
        self.train_dataset = self.train_bert_dataset.create(self.x_train, self.y_train_enc)

        self.val_bert_dataset = BertDataset(config.max_len, config.model_name, config.eval_batch_size)
        self.val_dataset = self.val_bert_dataset.create(self.x_val, self.y_val_enc)

        epochs = config.epochs
        for epoch in range(epochs):
            print(f"\nEpoch: {epoch+1}/{epochs}")
            self._train(self.config, self.model, self.loss_fn, self.optimizer)

            outputs, losses, preds, actuals = self.eval()
            actuals = [item.numpy() for sublist in actuals for item in sublist]
            preds = [item for sublist in preds for item in sublist]
            print(f"Validation Accuracy: {accuracy_score(np.argmax(actuals, axis=1).flatten(), preds)}")
            print(f"Validation MCC: {matthews_corrcoef(np.argmax(actuals, axis=1).flatten(), preds)}")
            print(f"Validation Loss: {np.mean(losses)}\n")
        return self

    def eval(self):
        outputs = []
        preds = []
        actuals = []
        losses = []

        @tf.function
        def eval_step(model, inputs, y):
            logits = model(inputs, training=False)
            loss = self.loss_fn(y, logits)
            return logits, loss, y

        for idx, (input_ids, label) in tqdm(enumerate(self.val_dataset)):
            logits, loss, y = eval_step(self.model, input_ids, label)
            outputs.append(logits)
            actuals.append(y)
            losses.append(loss)
            preds.append(np.argmax(logits, axis=1).flatten())
        return outputs, losses, preds, actuals

    def predict(self, test_dataset):
        preds = []

        @tf.function
        def pred_step(model, inputs):
            logits = model(inputs, training=False)
            return logits

        for idx, batch in tqdm(enumerate(test_dataset)):
            logits = pred_step(self.model, [batch[0], batch[1]])
            preds.append(np.argmax(logits, axis=1).flatten())
        return preds

    def create_submission(self):
        self.test_bert_dataset = BertDataset(self.config.max_len, self.config.model_name, self.config.eval_batch_size)
        self.test_dataset = self.test_bert_dataset.create(self.testdf["sentence"], [0, 1] * len(self.testdf))
        preds = self.predict(self.test_dataset)
        preds = [item for sublist in preds for item in sublist]
        self.testdf["Label"] = preds
        print(f"\n\nTest Data: \n{self.testdf['Label'].value_counts()}")
        self.testdf[["Id", "Label"]].to_csv("sample_submission.csv", index=False)
