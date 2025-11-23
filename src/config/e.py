from .config import Config


class Ep(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = False
        self.data.y_as_x = False
        self.data.filter_level = "X"
        self.data.channel_shift = 25


class Ef(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = True
        self.data.y_as_x = False
        self.data.filter_level = "X&Y"
        self.data.channel_shift = 0     # pervent mismatch between x and y


"""
Wait. But why we are retrain the whole transformer all the time? For any of our
test, the three optical channel will be the input. The only difference is if
we input BP as a extra channel or not and how we normalize the BP channel. 
Then, we can train a model with only three optical channels as input and 
a pretrain model and finetune it for different settings so that we don't need
to retrain the whole model all the time.
"""


class E01(Ep):
    def __init__(self):
        super().__init__()
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)
        self.trainer.max_epochs = 6350


"""
Make the pretrain task easier to make this pretriain more universal.
"""


class E02(Ep):
    def __init__(self):
        super().__init__()
        self.data.channel_drop = 0.4
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)
        self.runner.p_span_small = (0.0, 0.2)   # channel = 1
        self.runner.p_span_large = (0.2, 0.4)   # channel > 1
        self.trainer.max_epochs = 8208


"""
We finetune on regression on BP waveform without per-subject normalization.
This regression head surve as a global regression head for all subjects.
"""


class E03(Ef):
    def __init__(self):
        super().__init__()
        self.data.channel_drop = 0.2
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        self.runner.enable = (False, True, True)
        self.runner.weight = (0.0, 0.1, 0.9)
        self.runner.lr = 0.0005
        self.trainer.ckpt_load_path = \
            "ckpt/E02/epoch=3885-step=248704.ckpt"
