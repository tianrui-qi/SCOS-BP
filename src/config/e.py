from .config import Config


__all__ = [
    "ConfigE01", "ConfigE02", 
    "ConfigE03", "ConfigE04", 
    "ConfigE05", 
    "ConfigE06", "ConfigE07",
]


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


"""
Wait. But why we are retrain the whole transformer all the time? For any of our
test, the three optical channel will be the input. The only difference is if
we input BP as a extra channel or not and how we normalize the BP channel. 
Then, we can train a model with only three optical channels as input and 
a pretrain model and finetune it for different settings so that we don't need
to retrain the whole model all the time.
"""


class ConfigE05(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.y_as_channel = False
        self.data.filter_level = "X"
        self.trainer.max_epochs = 6350


"""
Make the pretrain task easier to make this pretriain more universal.
"""


class ConfigE06(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.y_as_channel = False
        self.data.filter_level = "X"
        self.data.channel_drop = 0.4
        self.runner.p_span_small = (0.0, 0.2)  # channel = 1
        self.runner.p_span_large = (0.2, 0.4)  # channel > 1
        self.trainer.max_epochs = 8208


"""
We finetune on regression on BP waveform without per-subject normalization.
This regression head surve as a global regression head for all subjects.
"""


class ConfigE07(ConfigE):
    def __init__(self):
        super().__init__()
        self.data.enable = (False, False, True, True)
        self.data.y_as_channel = False
        self.data.filter_level = "XY"
        self.data.channel_perm = True
        self.data.channel_drop = 0.2
        self.data.channel_shift = 0     # pervent mismatch between x and y
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        self.runner.enable = (False, False, True, True)
        self.runner.weight = (0.0, 0.0, 0.1, 0.9)
        self.runner.lr = 0.0005
        self.trainer.ckpt_load_path = \
            "ckpt/ConfigE06/epoch=3885-step=248704.ckpt"
