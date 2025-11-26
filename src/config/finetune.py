from .config import Config


class Finetune(Config):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        # disable randomness in finetune
        self.data.channel_perm = False
        self.data.channel_drop = 0
        self.data.channel_shift = 0
        # PyTorch uses the spawn start method on macOS/Windows
        # if a script creates a DataLoader with num_workers > 0 outside
        # an if __name__ == "__main__": guard, each worker will re-import the
        # main script and recursively launch new workers, causing a crash
        self.data.num_workers = 0
        # freeze parameters (except adapter)
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        self.runner.freeze_head = True
        self.runner.K = 50
        self.runner.weight_shape = 1e-4
        self.runner.weight_min = 0.5
        self.runner.weight_max = 0.5
        self.runner.lr = 0.1
        self.runner.step_size = 100
        self.runner.gamma = 0.1
        self.trainer.max_epochs = 1000
        self.trainer.ckpt_load_path = "ckpt/PretrainH/last.ckpt"
