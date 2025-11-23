from .config import Config


class Finetune(Config):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_perm = False
        self.data.channel_drop = 0.2
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        self.runner.freeze_head = True
        self.runner.weight_shape = 0.2
        self.runner.weight_min = 0.4
        self.runner.weight_max = 0.4
        self.runner.lr = 0.005
        self.runner.step_size = 30
        self.runner.gamma = 0.98
