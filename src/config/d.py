from .config import Config


__all__ = ["ConfigD02"]


class ConfigD01(Config):
    def __init__(self):
        super().__init__()
        self.data.x_load_path = "data/wave2wave/x.pt"
        self.data.y_load_path = "data/wave2wave/y.pt"
        self.data.split_load_path = "data/wave2wave/split02.pt"
        self.data.y_as_channel = True
        self.data.channel_drop = 0.5
        self.data.channel_shift = 25
        self.data.num_workers = 0
        self.model.S = 100
        self.model.stride = 25
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)
        self.runner.step_size = 30
        self.trainer.max_epochs = 2640


class ConfigD02(ConfigD01):
    def __init__(self):
        super().__init__()
        self.data.channel_drop = 1
        self.model.p_span_large = (0.5, 1.0)
        self.trainer.max_epochs = 10000
        self.trainer.ckpt_load_path = "ckpt/ConfigD01/last.ckpt"
        self.trainer.resume = True
