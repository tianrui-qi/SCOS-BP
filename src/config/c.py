from .config import Config


class ConfigCp(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = False
        self.data.y_as_x = False
        self.data.data_load_path = "data/wave2value.mat"
        self.data.split_type = "SubjectDependent"
        # [Contrastive, Reconstruction, Regression]
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)


class ConfigCf(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = True
        self.data.y_as_x = False
        self.data.data_load_path = "data/wave2value.mat"
        self.data.split_type = "SubjectDependent"
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        # [Contrastive, Reconstruction, Regression]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)


"""
We modify the reconstruction mask generate logic. 
We change the valid of regression task from use contrastive masking to no
masking.
Now, we perform ablation studies on stride.
"""


# S=100, stride=75


class ConfigC01(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.runner.step_size = 20


class ConfigC02(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.runner.lr = 0.0001
        self.runner.step_size = 50
        self.trainer.max_epochs = 5700
        self.trainer.ckpt_load_path = "ckpt/ConfigC01/last.ckpt"


class ConfigC03(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.runner.freeze_transformer = 4
        self.trainer.max_epochs = 7000
        self.trainer.ckpt_load_path = "ckpt/ConfigC02/last.ckpt"


class ConfigC04(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.trainer.max_epochs = 7000
        self.trainer.ckpt_load_path = "ckpt/ConfigC02/last.ckpt"


# S=100, stride=50


class ConfigC05(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 50


class ConfigC06(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 50
        self.runner.freeze_transformer = 4
        self.trainer.max_epochs = 7000
        self.trainer.ckpt_load_path = "ckpt/ConfigC05/last.ckpt"


class ConfigC07(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 50
        self.trainer.max_epochs = 7125
        self.trainer.ckpt_load_path = "ckpt/ConfigC05/last.ckpt"


# S=100, stride=25 


class ConfigC08(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 25


class ConfigC09(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 25
        self.trainer.max_epochs = 5500
        self.trainer.ckpt_load_path = "ckpt/ConfigC08/last.ckpt"


"""
Best Valid MSE Reconstruction: (before smoothing, after 0.9 smoothing)
-   ConfigC02, S=100, stride=75: (0.0357, 0.0386)
-   ConfigC05, S=100, stride=50: (0.0297, 0.0322)
-   ConfigC08, S=100, stride=25: (0.0244, 0.0266)
Best Valid MSE Regression: (before smoothing, after 0.9 smoothing)
-   ConfigC04, S=100, stride=75: (21.13, 22.19)
-   ConfigC07, S=100, stride=50: (18.89, 19.67)
-   ConfigC09, S=100, stride=25: (16.50, 18.07)
"""


# S=200, stride=50


class ConfigC10(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 50


class ConfigC11(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 50
        self.trainer.ckpt_load_path = "ckpt/ConfigC10/last.ckpt"


# S=200, stride=100


class ConfigC12(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 100


class ConfigC13(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 100
        self.trainer.ckpt_load_path = "ckpt/ConfigC12/last.ckpt"


# S=400, stride=50


class ConfigC14(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 50


class ConfigC15(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 50
        self.trainer.ckpt_load_path = "ckpt/ConfigC14/last.ckpt"


# S=400, stride=100


class ConfigC16(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 100


class ConfigC17(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 100
        self.trainer.ckpt_load_path = "ckpt/ConfigC16/last.ckpt"


# S=400, stride=200


class ConfigC18(ConfigCp):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 200


class ConfigC19(ConfigCf):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 200
        self.trainer.ckpt_load_path = "ckpt/ConfigC18/last.ckpt"


"""
Now we use the best configurations we have and train again on data split
where train and valid's subject are joint. 
"""


class ConfigC20(ConfigCp):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectIndependent"
        self.model.S = 100
        self.model.stride = 25


class ConfigC21(ConfigCf):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectIndependent"
        self.model.S = 100
        self.model.stride = 25
        self.trainer.ckpt_load_path = "ckpt/ConfigC20/last.ckpt"


"""
What if we only pretrain with joint subjects and finetune with disjoint
subjects?
"""


class ConfigC22(ConfigCf):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectIndependent"
        self.model.S = 100
        self.model.stride = 25
        self.trainer.ckpt_load_path = "ckpt/ConfigC08/last.ckpt"
