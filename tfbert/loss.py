from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore

binary_cross_entropy = BinaryCrossentropy(from_logits=True)


def bce(targets, logits):
    return binary_cross_entropy(targets, logits)
