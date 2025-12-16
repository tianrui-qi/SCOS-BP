We modify the reconstruction mask generate logic. 
We change the valid of regression task from use contrastive masking to no
masking.
Now, we perform ablation studies on stride.

S=100, stride=75

**[c/01](01.yaml)**
**[c/02](02.yaml)**
**[c/03](03.yaml)**
**[c/04](04.yaml)**

S=100, stride=50

**[c/05](05.yaml)**
**[c/06](06.yaml)**
**[c/07](07.yaml)**

S=100, stride=25 

**[c/08](08.yaml)**
**[c/09](09.yaml)**

Best Valid MSE Reconstruction: (before smoothing, after 0.9 smoothing)
-   ExpC02, S=100, stride=75: (0.0357, 0.0386)
-   ExpC05, S=100, stride=50: (0.0297, 0.0322)
-   ExpC08, S=100, stride=25: (0.0244, 0.0266)
Best Valid MSE Regression: (before smoothing, after 0.9 smoothing)
-   ExpC04, S=100, stride=75: (21.13, 22.19)
-   ExpC07, S=100, stride=50: (18.89, 19.67)
-   ExpC09, S=100, stride=25: (16.50, 18.07)

S=200, stride=50

**[c/10](10.yaml)**
**[c/11](11.yaml)**

S=200, stride=100

**[c/12](12.yaml)**
**[c/13](13.yaml)**

S=400, stride=50

**[c/14](14.yaml)**
**[c/15](15.yaml)**

S=400, stride=100

**[c/16](16.yaml)**
**[c/17](17.yaml)**

S=400, stride=200

**[c/18](18.yaml)**
**[c/19](19.yaml)**

Now we use the best configurations we have and train again on data split
where train and valid's measurement are joint. 

**[c/20](20.yaml)**
**[c/21](21.yaml)**

What if we only pretrain with joint measurements and finetune with disjoint
measurements?

**[c/22](22.yaml)**
