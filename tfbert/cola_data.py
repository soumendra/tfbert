import pandas as pd  # type: ignore
from pathlib import Path
from pandas import DataFrame
from typing import List, Union
from tensorflow.keras.utils import to_categorical  # type: ignore
import tensorflow_addons as tfa  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import tensorflow as tf  # type: ignore
from attrdict import AttrDict  # type: ignore

tf.get_logger().setLevel("INFO")


tokenizer = BertTokenizerFast("data/bert-base-uncased-vocab.txt", lowercase=True)


def preprocess(sentence, label, max_len):
    sentence = str(sentence.decode("utf-8"))
    sentence = " ".join(sentence.split())
    encoded = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return (input_ids, attention_mask, label)


class BertDataset(tf.data.Dataset):
    def _generator(sentences, labels, max_len):
        for sent, lbl in zip(sentences, labels):
            yield preprocess(sent, lbl, max_len)

    def __new__(cls, sentences, labels, max_len):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(sentences, labels, max_len),
            output_types=(tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.float32),
            output_shapes=((max_len,), (max_len,), (2,)),
        )

    @staticmethod
    def create(sentences, labels, max_len, batch_size):
        dataset = BertDataset(sentences, labels, max_len)
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size)
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

    def train(self, config: AttrDict, model, loss_fn, optimizer):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_dataset = BertDataset.create(self.x_train, self.y_train_enc, config.max_len, config.train_batch_size)
        self.val_dataset = BertDataset.create(self.x_val, self.y_val_enc, config.max_len, config.eval_batch_size)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["accuracy", tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)],
        )

        self.model.fit(
            self.train_dataset,
            epochs=config.epochs,
            validation_data=self.val_dataset,
            steps_per_epoch=len(self.x_train) // config.train_batch_size,
        )
        return self

    def create_submission(self):
        self.test_dataset = BertDataset.create(
            self.testdf["Sentence"].values,
            [[0, 1]] * len(self.testdf),  # creating fake labels
            self.config.max_len,
            self.config.eval_batch_size,
        )
        preds = self.model.predict(self.test_dataset)
        self.testdf["Label"] = preds
        print(f"\n\nTest Data: \n{self.testdf['Label'].value_counts()}")
        self.testdf[["Id", "Label"]].to_csv("sample_submission.csv", index=False)
