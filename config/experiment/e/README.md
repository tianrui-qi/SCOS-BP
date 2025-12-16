Wait. But why we are retrain the whole transformer all the time? For any of 
our test, the three optical channel will be the input. The only difference is 
if we input BP as a extra channel or not and how we normalize the BP 
channel. Then, we can train a model with only three optical channels as input 
and a pretrain model and finetune it for different settings so that we don't
need to retrain the whole model all the time.

**[`e/01`](01.yaml)**

Make the pretrain task easier to make this pretriain more universal.

**[`e/02`](02.yaml)**

We finetune on regression on BP waveform without per-measurement 
normalization. This regression head surve as a global regression head for all 
measurements.

**[`e/03`](03.yaml)**
