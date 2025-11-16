from .config import Config


__all__ = []


class ConfigA01(Config):
    def __init__(self):
        super().__init__()
        self.data.x_load_path = "data/wave2value/x.pt"
        self.data.y_load_path = "data/wave2value/y.pt"
        self.data.profile_load_path = "data/wave2value/profile.csv"
        # [Contrastive, ReconstructionRaw, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = (0.2,  0.8,  0.0)
        self.trainer.max_epochs = 2350


class ConfigA02(Config):
    def __init__(self):
        super().__init__()
        self.data.x_load_path = "data/wave2value/x.pt"
        self.data.y_load_path = "data/wave2value/y.pt"
        self.data.profile_load_path = "data/wave2value/profile.csv"
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        # [Contrastive, ReconstructionRaw, RegressionSingle]
        self.runner.enable = (True, True,    True)
        self.runner.weight = ( 0.2,  0.8, 0.00001)
        self.trainer.max_epochs = 2860
        self.trainer.ckpt_load_path = "ckpt/ConfigA01/last.ckpt"
        self.trainer.resume = True


class ConfigA03(Config):
    def __init__(self):
        super().__init__()
        self.data.x_load_path = "data/wave2value/x.pt"
        self.data.y_load_path = "data/wave2value/y.pt"
        self.data.profile_load_path = "data/wave2value/profile.csv"
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 2
        # [Contrastive, ReconstructionRaw, RegressionSingle]
        self.runner.enable = (True, True,   True)
        self.runner.weight = ( 0.4,  0.6, 0.0001)
        self.trainer.max_epochs = 5000
        self.trainer.ckpt_load_path = "ckpt/ConfigA02/last.ckpt"
        self.trainer.resume = True
