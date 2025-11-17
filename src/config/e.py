from .config import Config


__all__ = ["ConfigE01", "ConfigE02"]


class ConfigE(Config):
    def __init__(self):
        super().__init__()
        self.data.channel_shift = 25
        self.data.num_workers = 0
        self.model.S = 100
        self.model.stride = 25
        self.runner.step_size = 30


"""
Since name of reconstruction head is changed, ckpt will not be loaded correctly.
We load ckpt from ConfigD06 and train our new head here. Also as a verification
of our new pipeline.
"""


class ConfigE01(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.enable = (False, False, True, False)
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        self.runner.enable = (False, False, True, False)
        self.runner.weight = (  0.0,   0.0, 1.0,   0.0)
        self.runner.p_span_small = (0.0, 0.8)
        self.runner.p_span_large = (0.8, 1.0)
        self.runner.lr = 0.0005
        self.trainer.max_epochs = 170
        self.trainer.ckpt_load_path = "ckpt/ConfigD06/last.ckpt"


class ConfigE02(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.enable = (False, True, False, False)
        self.data.P = 32
        self.data.batch_size = 64
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        self.runner.enable = (False, True, False, False)
        self.runner.weight = (  0.0,  1.0,   0.0,   0.0)
        self.runner.p_span_small = (0.0, 0.8)
        self.runner.p_span_large = (0.8, 1.0)
        self.runner.lr = 0.001
        self.trainer.ckpt_load_path = "ckpt/ConfigE01/last.ckpt"
