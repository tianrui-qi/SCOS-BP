from .config import Config


__all__ = ["ConfigD04", "ConfigD05", "ConfigD06"]


class ConfigD(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_channel = True
        self.data.channel_shift = 25
        self.data.num_workers = 0
        self.model.S = 100
        self.model.stride = 25
        # [Contrastive, ReconstructionRaw, Regression]
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)
        self.runner.step_size = 30


"""
Now, we start to reconstruct BP waveform instead of regression BP values.
First, we validate on split02 (joint) to make sure everything works well.
"""


class ConfigD01(ConfigD):
    def __init__(self):
        super().__init__()
        self.data.split_load_path = "data/wave2wave/split02.pt"
        self.data.channel_drop = 0.5
        self.trainer.max_epochs = 2640


class ConfigD02(ConfigD):
    def __init__(self):
        super().__init__()
        self.data.split_load_path = "data/wave2wave/split02.pt"
        self.runner.p_span_large = (0.5, 1.0)
        self.trainer.ckpt_load_path = "ckpt/ConfigD01/last.ckpt"
        self.trainer.resume = True


"""
Now, we train on split01 (disjoint).
"""


class ConfigD03(ConfigD):
    def __init__(self):
        super().__init__()
        self.runner.p_span_large = (0.5, 1.0)


class ConfigD04(ConfigD):
    def __init__(self):
        super().__init__()
        self.data.channel_perm = False
        self.data.channel_drop = 0.5
        self.runner.p_point = 0.05
        self.runner.p_span_small = (0.3, 0.5)
        self.runner.p_span_large = (0.9, 1.0)
        self.runner.lr = 0.001
        self.trainer.ckpt_load_path = "ckpt/ConfigD03/last.ckpt"


"""
Try to solve calibration problem by put condition 1 data into training as well.
"""


class ConfigD05(ConfigD):
    def __init__(self):
        super().__init__()
        self.data.split_load_path = "data/wave2wave/split03.pt"
        self.data.channel_perm = False
        self.data.channel_drop = 0.5
        self.runner.p_point = 0.05
        self.runner.p_span_small = (0.3, 0.5)
        self.runner.p_span_large = (0.9, 1.0)
        self.runner.lr = 0.0008
        self.trainer.max_epochs = 5340
        self.trainer.ckpt_load_path = "ckpt/ConfigD04/last.ckpt"


"""
Not good. But it's seems like it overfit. So now, we start our training from
ConfigD03 again with more randomness still on split01. We save ckpt every 200 
epochs to see how training affects the performance.
"""


class ConfigD06(ConfigD):
    def __init__(self):
        super().__init__()
        self.runner.p_span_small = (0.0, 0.8)
        self.runner.p_span_large = (0.8, 1.0)
        self.runner.lr = 0.0005
        self.trainer.max_epochs = 7610
        self.trainer.ckpt_load_path = "ckpt/ConfigD03/last.ckpt"
