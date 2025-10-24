from .config import Config


__all__ = []


class ConfigA01(Config):
    def __init__(self):
        super().__init__()
        self.data["profile_load_path"] = "data/waveform/profile.csv"
        self.trainer["max_epochs"] = 2350


class ConfigA02(Config):
    def __init__(self):
        super().__init__()
        self.data["profile_load_path"] = "data/waveform/profile.csv"
        self.model["freeze_embedding"] = True
        self.model["freeze_transformer"] = 3
        self.runner["weight"] = [0.2, 0.8, 0.00001]
        self.trainer["max_epochs"] = 2860
        self.trainer["ckpt_load_path"] = "ckpt/ConfigA01/last.ckpt"
        self.trainer["resume"] = True


class ConfigA03(Config):
    def __init__(self):
        super().__init__()
        self.data["profile_load_path"] = "data/waveform/profile.csv"
        self.model["freeze_embedding"] = True
        self.model["freeze_transformer"] = 2
        self.runner["weight"] = [0.4, 0.6, 0.0001]
        self.trainer["max_epochs"] = 5000
        self.trainer["ckpt_load_path"] = "ckpt/ConfigA02/last.ckpt"
        self.trainer["resume"] = True
