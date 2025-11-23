from .config import Config


class Finetune(Config):
    def __init__(self):
        super().__init__()
        # TODO: find best hyperparameters for finetuning
        self.data.filter_level = "All"
        self.data.channel_perm = False
        self.data.channel_drop = 0
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        self.runner.freeze_head = True
        self.runner.K = 50
        self.runner.weight_shape = 0.1
        self.runner.weight_min = 0.45
        self.runner.weight_max = 0.45
        self.runner.lr = 0.1
        self.runner.step_size = 30
        self.runner.gamma = 0.98
        self.trainer.every_n_epochs = 500
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"
