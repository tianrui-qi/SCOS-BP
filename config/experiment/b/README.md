**[`b/01`](01.yaml)**

First load the last checkpoint and train with higher weight on 
contrastive, see if the loss go down. maybe it goes up since focus too 
much on reconstruction.

**[`b/02`](02.yaml)**

No, both train and valid loss go down and then go up again. also, the
performance of contrastive is better then `b/01` but reconstruction is 
worse.
Maybe try mutiloss in the future. do not open that at begin. open 
mutiloss when the performance of both tasks are good enough. then check 
if loss go up situation still happen or not.
Also, maybe try two way backpropagation for contrastive loss, i.e.,
do not detach z_orig.
Now, we finetune with regression task.

**[`b/03`](03.yaml)**
**[`b/04`](04.yaml)**

Since previous bias split not good enough, we not sure if this bad 
performance is because of model or becuase of data. So, now with split02,
each measurement's samples appear in both train and valid set. There is no 
bias in data split. We expect the model's train and valid loss of regression
perform similarly. To do so, we first pretrain the model with only
reconstruction task. We do not use contrastive task to reduce complexity
and control variables.

**[`b/05`](05.yaml)**
**[`b/06`](06.yaml)**

After pretrain only with reconstruction, we finetune by regression task.

**[`b/07`](07.yaml)**
**[`b/08`](08.yaml)**
**[`b/09`](09.yaml)**

Notes that from `b/07` to `b/09`, as we open more layers of model to
train, the learning speed increase and performance improve. In `b/08`,
valid loss go to stage at around MSE=100, while in `b/09`, valid loss
reach MSE=70 while training not finished. So, maybe current single output
linear layer regression head is not complex enough. We implement a deeper 
regession head with 2 hidden linear layers and 1 output linear layer. We
want to see how it perform.
for enable and weight, now we have four tasks and the order is:
-   Contrastive
-   Reconstruction
-   RegressionSingle
-   RegressionDeep

**[`b/10`](10.yaml)**

The loss go down very slow. In previous experiments, we notes that if we
train contrastive and reconstruction tasks together, the model learn
faster. So here, to increase speed of training, first we train regression
and reconstruction tasks together; second from the observation of `b/07`
to `b/09`, we open more layers to train.

**[`b/11`](11.yaml)**

Performance much better. Now performance of train and valid are very close.
Previously, valid loss always go to stage while train loss keep going down.
However, we change many setting this time: 
1.  add reconstruction task,
2.  more layers open to train,
3.  deeper regression head.
To check which factor cause the improvement, we perform a ablation study.

**[`b/12`](12.yaml)**
**[`b/13`](13.yaml)**

1.  add reconstruction task
-   Compare `b/13-1` and `b/11`, they perform nearly the same. The only
    difference is `b/11` has reconstruction task while `b/13-1` not. 
+   Thus, adding reconstruction task may not help much.
2.  more layers open to train
-   From `b/13-0`, `b/13-1`, `b/13-2`, `b/13-3`, `b/13-4`, the number 
    of layers open to train decrease from 4 to 0. There is little 
    performance diff except `b/13-4` where transformer all frozen. Also, 
    less layers open to train lead to a slightly better performance in valid.
    However, this may because less parameters lead to faster convergence 
    within same epochs.
+   Thus, for finetuning stage, we can leave last layer of transformer 
    trainable and freeze front layers. After model converge, we can 
    unfreeze one more layer to see if performance improve.
3.  deeper regression head
-   Compare `b/11` and `b/12`, their diff is regression head. We notice
    that both train loss go down similarly to around MSE=25. However, 
    `b/11`'s valid MSE=36 where `b/12`'s valid MSE=60. Althogh they 
    load diff checkpoints, these checkpoints perform similar where MSE=100.
+   Thus, a deeper regression head help improve performance.
-   Compare `b/09` and `b/12`, since two experiments above show 
    reconstruction task and number of trainable layers have little effect,
    the main diff between them is batch size: `b/09` has batch size 1024 
    while `b/12` has batch size 256. `b/09`'s train loss go down to
    MSE=36, similar to `b/11` and `b/12` but valid loss go to MSE=75,
    much worse than `b/12`'s MSE=60. 
+   This may suggest that larger batch size lead to worse generalization 
    performance.
According to hypothesis from 3.2, we perform ablation study on batch size. 
We set all other config to the best config found in previous experiments,
i.e., `b/13-3`.

**[`b/14`](14.yaml)**

We focus on when train reach same loss, how is the valid loss.
-   `b/14-0`: batch=32,   unstable
-   `b/14-1`: batch=128,  train=41.4025, valid=45.7383
-   `b/13-3`: batch=256,  train=41.5971, valid=46.6691
-   `b/14-2`: batch=512,  train=41.5638, valid=47.4214
-   `b/14-3`: batch=2048, train=41.6944, valid=49.3266
Note that these values read from tensorboard with smooth=0.9.
We find that smaller batch size lead to better generalization performance,
which is consistent with our hypothesis. Current default batch size is good
enough since smaller batch size lead to unstable training.
Now we finish the training on best config found, `b/13-3`, as base line of
future experiments.

**[`b/15`](15.yaml)**

Now, instead of only train regression head, we also train transformer during
finetuning. Then, we are interested in how pretrain contribute to final
performance. Thus, we train the model direclty with regression task, without
pretrain.

**[`b/16`](16.yaml)**
**[`b/17`](17.yaml)**

Training is much slower than without pretrain. Since we use lr schedular and
training is too slow, not sure if not converge cause by lr too slow. But we
can see pretrain is useful.
