from .config import Config


__all__ = ["ConfigE01", "ConfigE02", "ConfigE03", "ConfigE04"]


class ConfigE(Config):
    def __init__(self):
        super().__init__()
        self.data.enable = (False, False, True, False)
        self.data.channel_shift = 25
        self.runner.enable = (False, False, True, False)
        self.runner.weight = (  0.0,   0.0, 1.0,   0.0)
        self.model.S = 100
        self.model.stride = 25


class ConfigE01(ConfigE):
    def __init__(self):
        super().__init__()
        self.runner.p_span_large = (0.5, 1.0)
        self.runner.lr = 0.003
        self.trainer.max_epochs = 5000


class ConfigE02(ConfigE):
    def __init__(self):
        super().__init__()
        self.runner.p_point = 0.05
        self.runner.p_span_small = (0.0, 0.8)
        self.runner.p_span_large = (0.8, 1.0)
        self.runner.lr = 0.001
        self.trainer.ckpt_load_path = "ckpt/ConfigE01/last.ckpt"
        self.trainer.max_epochs = 5000


class ConfigE03(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.mu = None     # per-subject mu and global sd
        self.runner.p_span_large = (0.5, 1.0)
        self.runner.lr = 0.003
        self.trainer.max_epochs = 5000


class ConfigE04(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.mu = None     # per-subject mu and global sd
        self.runner.p_point = 0.05
        self.runner.p_span_small = (0.0, 0.8)
        self.runner.p_span_large = (0.8, 1.0)
        self.runner.lr = 0.001
        self.trainer.ckpt_load_path = "ckpt/ConfigE03/last.ckpt"
        self.trainer.max_epochs = 5000
