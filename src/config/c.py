from .config import Config


class Cp(Config):
    def __init__(self):
        super().__init__()
        self.data.y_as_y = False
        self.data.y_as_x = False
        self.data.data_load_path = "data/wave2value.mat"
        self.data.split_type = "SubjectDependent"
        # [Contrastive, Reconstruction, Regression]
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)


class Cf(Config):
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


class C01(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.runner.step_size = 20


class C02(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.runner.lr = 0.0001
        self.runner.step_size = 50
        self.trainer.max_epochs = 5700
        self.trainer.ckpt_load_path = "ckpt/C01/last.ckpt"


class C03(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.runner.freeze_transformer = 4
        self.trainer.max_epochs = 7000
        self.trainer.ckpt_load_path = "ckpt/C02/last.ckpt"


class C04(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 75
        self.trainer.max_epochs = 7000
        self.trainer.ckpt_load_path = "ckpt/C02/last.ckpt"


# S=100, stride=50


class C05(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 50


class C06(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 50
        self.runner.freeze_transformer = 4
        self.trainer.max_epochs = 7000
        self.trainer.ckpt_load_path = "ckpt/C05/last.ckpt"


class C07(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 50
        self.trainer.max_epochs = 7125
        self.trainer.ckpt_load_path = "ckpt/C05/last.ckpt"


# S=100, stride=25 


class C08(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 25


class C09(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 100
        self.model.stride = 25
        self.trainer.max_epochs = 5500
        self.trainer.ckpt_load_path = "ckpt/C08/last.ckpt"


"""
Best Valid MSE Reconstruction: (before smoothing, after 0.9 smoothing)
-   C02, S=100, stride=75: (0.0357, 0.0386)
-   C05, S=100, stride=50: (0.0297, 0.0322)
-   C08, S=100, stride=25: (0.0244, 0.0266)
Best Valid MSE Regression: (before smoothing, after 0.9 smoothing)
-   C04, S=100, stride=75: (21.13, 22.19)
-   C07, S=100, stride=50: (18.89, 19.67)
-   C09, S=100, stride=25: (16.50, 18.07)
"""


# S=200, stride=50


class C10(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 50


class C11(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 50
        self.trainer.ckpt_load_path = "ckpt/C10/last.ckpt"


# S=200, stride=100


class C12(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 100


class C13(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 200
        self.model.stride = 100
        self.trainer.ckpt_load_path = "ckpt/C12/last.ckpt"


# S=400, stride=50


class C14(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 50


class C15(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 50
        self.trainer.ckpt_load_path = "ckpt/C14/last.ckpt"


# S=400, stride=100


class C16(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 100


class C17(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 100
        self.trainer.ckpt_load_path = "ckpt/C16/last.ckpt"


# S=400, stride=200


class C18(Cp):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 200


class C19(Cf):
    def __init__(self):
        super().__init__()
        self.model.S = 400
        self.model.stride = 200
        self.trainer.ckpt_load_path = "ckpt/C18/last.ckpt"


"""
Now we use the best configurations we have and train again on data split
where train and valid's subject are joint. 
"""


class C20(Cp):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectIndependent"
        self.model.S = 100
        self.model.stride = 25


class C21(Cf):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectIndependent"
        self.model.S = 100
        self.model.stride = 25
        self.trainer.ckpt_load_path = "ckpt/C20/last.ckpt"


"""
What if we only pretrain with joint subjects and finetune with disjoint
subjects?
"""


class C22(Cf):
    def __init__(self):
        super().__init__()
        self.data.split_type = "SubjectIndependent"
        self.model.S = 100
        self.model.stride = 25
        self.trainer.ckpt_load_path = "ckpt/C08/last.ckpt"
