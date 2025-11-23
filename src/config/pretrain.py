from .config import Config


class PretrainT(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = False
        self.data.filter_level = "X"
        self.data.channel_drop = 0.4
        self.data.channel_shift = 25
        self.runner.p_span_small = (0.0, 0.2)   # channel = 1
        self.runner.p_span_large = (0.2, 0.4)   # channel > 1
        self.trainer.max_epochs = 4000


class PretrainH(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = True
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        self.runner.enable = (False, True, True)
        self.runner.weight = (0.0, 0.1, 0.9)
        self.runner.lr = 0.0005
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainT/epoch=3885-step=248704.ckpt"
