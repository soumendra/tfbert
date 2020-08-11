from attrdict import AttrDict  # type: ignore

configs_local = {
    "random_state": 42,
    "max_len": 64,
    "train_batch_size": 32,
    "eval_batch_size": 8,
    "epochs": 3,
    "lr": 3e-5,
    "datapath": "data/cola_public_1.1/cola_public/raw/",
}

configs_local = AttrDict(configs_local)
