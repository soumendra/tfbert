from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore

binary_cross_entropy = BinaryCrossentropy()


def bce(targets, logits):
    return binary_cross_entropy(targets, logits)
