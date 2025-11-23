from .config import Config


__all__ = []


class D(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = True
        self.data.y_as_x = True
        self.data.channel_shift = 25
        # [Contrastive, Reconstruction, Regression]
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)


"""
Now, we start to reconstruct BP waveform instead of regression BP values.
First, we validate on split02 (joint) to make sure everything works well.
"""


class D01(D):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectDependent"
        self.data.channel_drop = 0.5
        self.trainer.max_epochs = 2640


class D02(D):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectDependent"
        self.runner.p_span_large = (0.5, 1.0)
        self.trainer.ckpt_load_path = "ckpt/D01/last.ckpt"
        self.trainer.resume = True


"""
Now, we train on split01 (disjoint).
"""


class D03(D):
    def __init__(self):
        super().__init__()
        self.runner.p_span_large = (0.5, 1.0)


class D04(D):
    def __init__(self):
        super().__init__()
        self.data.channel_perm = False
        self.data.channel_drop = 0.5
        self.runner.p_point = 0.05
        self.runner.p_span_small = (0.3, 0.5)
        self.runner.p_span_large = (0.9, 1.0)
        self.runner.lr = 0.001
        self.trainer.ckpt_load_path = "ckpt/D03/last.ckpt"


"""
Try to solve calibration problem by put condition 1 data into training as well.
"""


class D05(D):
    def __init__(self):
        super().__init__()
        self.data.split_type = "split03"    # type: ignore # not support
        self.data.channel_perm = False
        self.data.channel_drop = 0.5
        self.runner.p_point = 0.05
        self.runner.p_span_small = (0.3, 0.5)
        self.runner.p_span_large = (0.9, 1.0)
        self.runner.lr = 0.0008
        self.trainer.max_epochs = 5340
        self.trainer.ckpt_load_path = "ckpt/D04/last.ckpt"


"""
Not good. But it's seems like it overfit. So now, we start our training from
D03 again with more randomness still on split01. We save ckpt every 200 
epochs to see how training affects the performance.
"""


class D06(D):
    def __init__(self):
        super().__init__()
        self.runner.p_span_small = (0.0, 0.8)
        self.runner.p_span_large = (0.8, 1.0)
        self.runner.lr = 0.0005
        self.trainer.max_epochs = 7610
        self.trainer.ckpt_load_path = "ckpt/D03/last.ckpt"
