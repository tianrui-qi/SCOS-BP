from .config import Config


class B01(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = ( 0.2,  0.8,  0.0)
        self.runner.lr = 0.001
        self.runner.step_size = 20
        self.trainer.max_epochs = 4240


"""
First load the last checkpoint and train with higher weight on 
contrastive, see if the loss go down. maybe it goes up since focus too 
much on reconstruction.
"""


class B02(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = ( 0.5,  0.5,  0.0)
        self.runner.lr = 0.0005
        self.runner.step_size = 20
        self.trainer.max_epochs = 4055
        self.trainer.ckpt_load_path = \
            "ckpt/B01/last.ckpt"


"""
No, both train and valid loss go down and then go up again. also, the
performance of contrastive is better then B01 but reconstruction is 
worse.
Maybe try mutiloss in the future. do not open that at begin. open 
mutiloss when the performance of both tasks are good enough. then check 
if loss go up situation still happen or not.
Also, maybe try two way backpropagation for contrastive loss, i.e.,
do not detach z_orig.
Now, we finetune with regression task.
"""


class B03(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = ( 0.0,  0.0,  1.0)
        self.runner.lr = 0.0001
        self.runner.step_size = 20
        self.trainer.max_epochs = 2105
        self.trainer.ckpt_load_path = \
            "ckpt/B01/epoch=1162-step=46520.ckpt"


class B04(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = ( 0.0,  0.0,  1.0)
        self.runner.lr = 0.0001
        self.runner.step_size = 20
        self.trainer.max_epochs = 3470
        self.trainer.ckpt_load_path = \
            "ckpt/B03/last.ckpt"


"""
Since previous bias split not good enough, we not sure if this bad 
performance is because of model or becuase of data. So, now with split02,
each subject's samples appear in both train and valid set. There is no bias
in data split. We expect the model's train and valid loss of regression
perform similarly. To do so, we first pretrain the model with only
reconstruction task. We do not use contrastive task to reduce complexity
and control variables.
"""


class B05(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (True, True, True)
        self.runner.weight = ( 0.0,  1.0,  0.0)
        self.runner.step_size = 20
        self.trainer.max_epochs = 3620


class B06(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.data.batch_size = 1024
        self.data.num_workers = 16
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (False, True, False)
        self.runner.weight = (  0.0,  1.0,   0.0)
        self.runner.lr = 0.0005
        self.runner.step_size = 20
        self.trainer.max_epochs = 4861
        self.trainer.ckpt_load_path = \
            "ckpt/B05/last.ckpt"


"""
After pretrain only with reconstruction, we finetune by regression task.
"""


class B07(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.data.batch_size = 1024
        self.data.num_workers = 16
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.step_size = 20
        self.trainer.ckpt_load_path = \
            "ckpt/B06/epoch=1004-step=10050.ckpt"


class B08(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.data.batch_size = 1024
        self.data.num_workers = 16
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.lr = 0.001
        self.runner.step_size = 20
        self.trainer.max_epochs = 3205
        self.trainer.ckpt_load_path = \
            "ckpt/B07/last.ckpt"


class B09(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.data.batch_size = 1024
        self.data.num_workers = 16
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 2
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.lr = 0.001
        self.runner.step_size = 20
        self.trainer.max_epochs = 490
        self.trainer.ckpt_load_path = \
            "ckpt/B08/last.ckpt"


"""
Notes that from B07 to B09, as we open more layers of model to
train, the learning speed increase and performance improve. In B08,
valid loss go to stage at around MSE=100, while in B09, valid loss
reach MSE=70 while training not finished. So, maybe current single output
linear layer regression head is not complex enough. We implement a deeper 
regession head with 2 hidden linear layers and 1 output linear layer. We
want to see how it perform.
for enable and weight, now we have four tasks and the order is:
-   Contrastive
-   Reconstruction
-   RegressionSingle
-   RegressionDeep
"""


class B10(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.data.batch_size = 1024
        self.data.num_workers = 16
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4
        # [Contrastive, Reconstruction, RegressionDeep]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.lr = 0.01
        self.runner.step_size = 20
        self.trainer.max_epochs = 3195
        self.trainer.ckpt_load_path = \
            "ckpt/B06/epoch=1004-step=10050.ckpt"


"""
The loss go down very slow. In previous experiments, we notes that if we
train contrastive and reconstruction tasks together, the model learn
faster. So here, to increase speed of training, first we train regression
and reconstruction tasks together; second from the observation of B07 
to B09, we open more layers to train.
"""


class B11(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 1
        # [Contrastive, Reconstruction, RegressionDeep]
        self.runner.enable = (False, True, True)
        self.runner.weight = (  0.0,  1.0,  1.0)
        self.runner.step_size = 20
        self.trainer.max_epochs = 425
        self.trainer.ckpt_load_path = \
            "ckpt/B10/last.ckpt"


"""
Performance much better. Now performance of train and valid are very close.
Previously, valid loss always go to stage while train loss keep going down.
However, we change many setting this time: 
1.  add reconstruction task,
2.  more layers open to train,
3.  deeper regression head.
To check which factor cause the improvement, we perform a ablation study.
"""


class B12(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 1
        # [Contrastive, Reconstruction, RegressionSingle]
        self.runner.enable = (False, True, True)
        self.runner.weight = (  0.0,  1.0,  1.0)
        self.runner.step_size = 20
        self.trainer.max_epochs = 434
        self.trainer.ckpt_load_path = \
            "ckpt/B08/last.ckpt"


class B13(B11):
    def __init__(self):
        super().__init__()
        self.runner.freeze_transformer = 1
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)


class B14(B11):
    def __init__(self):
        super().__init__()
        self.runner.freeze_transformer = 0
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)


class B15(B11):
    def __init__(self):
        super().__init__()
        self.runner.freeze_transformer = 2
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)


class B16(B11):
    def __init__(self):
        super().__init__()
        self.runner.freeze_transformer = 3
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)


class B17(B11):
    def __init__(self):
        super().__init__()
        self.runner.freeze_transformer = 4
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)


""""
1.  add reconstruction task
-   Compare B13 and B11, they perform nearly the same. The only
    difference is B11 has reconstruction task while B13 not. 
+   Thus, adding reconstruction task may not help much.
2.  more layers open to train
-   From B14, B13, B15, B16, B17, the number 
    of layers open to train decrease from 4 to 0. There is little 
    performance diff except B17 where transformer all frozen. Also, 
    less layers open to train lead to a slightly better performance in valid.
    However, this may because less parameters lead to faster convergence 
    within same epochs.
+   Thus, for finetuning stage, we can leave last layer of transformer 
    trainable and freeze front layers. After model converge, we can 
    unfreeze one more layer to see if performance improve.
3.  deeper regression head
-   Compare B11 and B12, their diff is regression head. We notice
    that both train loss go down similarly to around MSE=25. However, 
    B11's valid MSE=36 where B12's valid MSE=60. Althogh they 
    load diff checkpoints, these checkpoints perform similar where MSE=100.
+   Thus, a deeper regression head help improve performance.
-   Compare B09 and B12, since two experiments above show 
    reconstruction task and number of trainable layers have little effect,
    the main diff between them is batch size: B09 has batch size 1024 
    while B12 has batch size 256. B09's train loss go down to
    MSE=36, similar to B11 and B12 but valid loss go to MSE=75,
    much worse than Config12's MSE=60. 
+   This may suggest that larger batch size lead to worse generalization 
    performance.
According to hypothesis from 3.2, we perform ablation study on batch size. 
We set all other config to the best config found in previous experiments,
i.e., B16.
"""


class B18(B16):
    def __init__(self):
        super().__init__()
        self.data.batch_size = 2048
        self.trainer.max_epochs = 500


class B19(B16):
    def __init__(self):
        super().__init__()
        self.data.batch_size = 512
        self.trainer.max_epochs = 500


class B20(B16):
    def __init__(self):
        super().__init__()
        self.data.batch_size = 128
        self.trainer.max_epochs = 500


class B21(B16):
    def __init__(self):
        super().__init__()
        self.data.batch_size = 32
        self.trainer.max_epochs = 500


"""
We focus on when train reach same loss, how is the valid loss.
-   B18: batch=2048, train=41.6944, valid=49.3266
-   B19: batch=512,  train=41.5638, valid=47.4214
-   B16: batch=256,  train=41.5971, valid=46.6691
-   B20: batch=128,  train=41.4025, valid=45.7383
-   B21: batch=32,   unstable
Note that these values read from tensorboard with smooth=0.9.
We find that smaller batch size lead to better generalization performance,
which is consistent with our hypothesis. Current default batch size is good
enough since smaller batch size lead to unstable training.
Now we finish the training on best config found, B16, as base line of
future experiments.
"""


class B22(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.model.S = 40
        self.model.stride = 20
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 3
        # [Contrastive, Reconstruction, RegressionDeep]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.step_size = 20
        self.trainer.ckpt_load_path = \
            "ckpt/B16/last.ckpt"
        self.trainer.resume = True


"""
Now, instead of only train regression head, we also train transformer during
finetuning. Then, we are interested in how pretrain contribute to final
performance. Thus, we train the model direclty with regression task, without
pretrain.
"""


class B23(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionDeep]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.step_size = 20
        self.trainer.max_epochs = 7580


class B24(Config):
    def __init__(self):
        super().__init__()
        self.data.data_load_path = "data/wave2value.mat"
        self.data.y_as_x = False
        self.data.split_type = "SubjectDependent"
        self.model.S = 40
        self.model.stride = 20
        # [Contrastive, Reconstruction, RegressionDeep]
        self.runner.enable = (False, False, True)
        self.runner.weight = (  0.0,   0.0,  1.0)
        self.runner.step_size = 20
        self.trainer.ckpt_load_path = \
            "ckpt/B23/last.ckpt"


"""
Training is much slower than without pretrain. Since we use lr schedular and
training is too slow, not sure if not converge cause by lr too slow. But we
can see pretrain is useful.
"""
