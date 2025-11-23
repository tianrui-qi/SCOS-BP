from .config import Config


class ConfigA01(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_y = True
        self.data.y_as_x = False
        self.data.split_type = "split"  # type: ignore # not support anymore
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = (0.2,  0.8,  0.0)
        self.runner.step_size = 20
        self.trainer.max_epochs = 2350


class ConfigA02(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_y = True
        self.data.y_as_x = False
        self.data.split_type = "split"  # type: ignore # not support anymore
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True,    True)
        self.runner.weight = ( 0.2,  0.8, 0.00001)
        self.runner.step_size = 20
        self.trainer.max_epochs = 2860
        self.trainer.ckpt_load_path = "ckpt/ConfigA01/last.ckpt"
        self.trainer.resume = True


class ConfigA03(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_y = True
        self.data.y_as_x = False
        self.data.split_type = "split"  # type: ignore # not support anymore
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 2
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True,   True)
        self.runner.weight = ( 0.4,  0.6, 0.0001)
        self.runner.step_size = 20
        self.trainer.max_epochs = 5000
        self.trainer.ckpt_load_path = "ckpt/ConfigA02/last.ckpt"
        self.trainer.resume = True
