Now, we start to reconstruct BP waveform instead of regression BP values.
First, we validate on split02 (joint) to make sure everything works well.

**[`d/01`](01.yaml)**
**[`d/02`](02.yaml)**

Now, we train on split01 (disjoint).

**[`d/03`](03.yaml)**
**[`d/04`](04.yaml)**

Try to solve calibration problem by put condition 1 data into training as 
well.

**[`d/05`](05.yaml)**

Not good. But it's seems like it overfit. So now, we start our training from
`d/03` again with more randomness still on split01. We save ckpt every 200 
epochs to see how training affects the performance.

**[`d/06`](06.yaml)**
