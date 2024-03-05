sweep_config = {
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "lr_gen": {"values": [0.00005, 0.0001, 0.0002, 0.0005]},
        "lr_disc": {"values": [0.00005, 0.0001, 0.0002, 0.0005]},
        "weight_init": {"values": ["normal", "xavier", "kaiming"]},
        "loss_type": {"values": ["BCE", "LSGAN", "Hinge"]},
        "optimizer_type": {"values": ["adam", "sgd", "rmsprop"]},
    },
}
