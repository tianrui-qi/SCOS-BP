from .config import Config


class F(Config):
    def __init__(self):
        super().__init__()
        self.data.channel_perm = False
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        self.runner.freeze_head = True
        self.runner.K = 50
        self.runner.step_size = 30
        self.runner.gamma = 0.98
        self.trainer.every_n_epochs = 500


class F01(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"


class F02(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"
        

class F03(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"
        

class F04(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"


class F05(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"


class F06(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"
        

class F07(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"
        

class F08(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.20
        self.runner.weight_min = 0.40
        self.runner.weight_max = 0.40
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"


class F09(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"


class F10(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"
        

class F11(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"
        

class F12(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.1
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"


class F13(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"


class F14(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "All"
        self.data.channel_drop = 0
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"
        

class F15(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/epoch=2799.ckpt"
        

class F16(F):
    def __init__(self):
        super().__init__()
        self.data.filter_level = "X&Y"
        self.data.channel_drop = 0.2
        self.runner.weight_shape = 0.00
        self.runner.weight_min = 0.50
        self.runner.weight_max = 0.50
        self.runner.lr = 0.01
        self.trainer.ckpt_load_path = \
            "ckpt/PretrainH/last.ckpt"
