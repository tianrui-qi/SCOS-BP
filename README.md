Please check the project 
[presentation](https://cdn.jsdelivr.net/gh/tianrui-qi/SCOS-BP@main/asset/presentation.pdf)
for a quick overview. 

## Installation

This project is built with
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) 
(`lightning=2.5`) for deep learning model development and 
[Hydra](https://github.com/facebookresearch/hydra) (`hydra-core=1.3`) for 
configuration management. 
Familiarity with these frameworks is recommended for further development.
Python dependencies are managed with
[Conda](https://docs.conda.io/en/latest/).
To set up the environment,

```bash
# clone the repository
git clone git@github.com:tianrui-qi/SCOS-BP.git
cd SCOS-BP
# create the conda environment
conda env create -f environment.yaml
conda activate scos-bp
```

All demonstrations in this README are based on data and pretrained model 
checkpoints available on [OSF](https://osf.io/yqpht/). 
You can download them from command line as follows

```bash
# clone the OSF storage
osf --project yqpht clone
# merge OSF storage into the project root
rsync -av --progress yqpht/osfstorage/ ./
rm -r yqpht
```

If command `rsync` is not available on your system, you may use 
`mv yqpht/osfstorage/* ./` instead but less safe. Make sure understand what
these commands do before running them.

We provide a sanity-check [script](script/sanity.py) to verify 
environment and data are correctly set up. This script trains the model on a 
single fixed batch for few steps using the reconstruction objective. 
To run the script,

```bash
python -m script.sanity
```

The loss printed out should decrease over time and final plot should show 
reconstruction starting to fit input. 
In addition, this script is configured by 
[`config/pipeline/sanity.yaml`](config/pipeline/sanity.yaml) and supports 
command line overrides via
[Hydra](https://github.com/facebookresearch/hydra)'s syntax.
It's a good starting point to get familiar with configuration system of this 
project.

## Data

For data downloaded from [OSF](https://osf.io/yqpht/) in
[Installation](#installation), three files are provided under `data/raw/`:

-   `x.npy`
    Optical waveforms (33635 samples × 3 channels × 1000 time points), 
    stored as `float32`. 
    Channels including Finger 808nm BFi, Finger 808nm PPG, Wrist 808nm BFi.
-   `y.npy`
    Blood pressure (BP) waveforms (33635 samples × 1000 time points), 
    stored as `float32`.
-   `profile.csv`
    Metadata for each sample.

The figure below illustrates sample preparation process from raw
measurements. The bottom-right panel shows two example samples: the first
sample passes quality control for all waveforms, while the second sample
passes quality control for only two optical waveforms. Samples are retained
even when some channels are missing; in such cases, missing channels are
represented as NaNs in `x.npy` and `y.npy`.

![DataPreparation](/asset/DataPreparation.jpg)

A brief preview of `profile.csv`:

```python
import pandas as pd
profile = pd.read_csv("data/raw/profile.csv")
print(profile)
```

|       | subject   | group        | health   | system   | age   | measurement   | repeat   | arm   | pulse   | pulse_norm   | condition   | systole   | diastole   |
|------:|-----------|--------------|----------|----------|-------|---------------|----------|-------|--------:|-------------:|------------:|----------:|-----------:|
| 0     | S001      | original     | True     | False    | 28    | S001          | False    | False | 0       | 0.0000       | 1           | nan       | nan        |
| 1     | S001      | original     | True     | False    | 28    | S001          | False    | False | 1       | 0.0023       | 1           | nan       | nan        |
| 2     | S001      | original     | True     | False    | 28    | S001          | False    | False | 2       | 0.0045       | 1           | 138.2190  | 92.0960    |
| 3     | S001      | original     | True     | False    | 28    | S001          | False    | False | 3       | 0.0068       | 1           | 139.8688  | 91.0879    |
| ...   | ...       | ...          | ...      | ...      | ...   | ...           | ...      | ...   | ...     | ...          | ...         | ...       | ...        |
| 33631 | H006      | hypertensive | False    | True     | 52    | H006_R        | True     | True  | 207     | 0.9857       | 6           | 126.9918  | 103.4106   |
| 33632 | H006      | hypertensive | False    | True     | 52    | H006_R        | True     | True  | 208     | 0.9905       | 6           | 126.5592  | 103.0878   |
| 33633 | H006      | hypertensive | False    | True     | 52    | H006_R        | True     | True  | 209     | 0.9952       | 6           | nan       | nan        |
| 33634 | H006      | hypertensive | False    | True     | 52    | H006_R        | True     | True  | 210     | 1.0000       | 6           | nan       | nan        |

Columns `subject` to `age` are subject-level metadata,
`measurement` to `arm` are measurement-level metadata, and
`pulse` to `diastole` are sample-level metadata.
Some samples (13191/33635) miss systolic/diastolic values due to
quality control filtering.
Several columns are derived from others for convenience:

```python
profile["subject"] = profile["measurement"].str.split("_").str[0]
profile["health"] = profile["group"] != "hypertensive"
profile["system"] = profile["group"] != "original"
profile["pulse"] = profile.groupby("measurement").cumcount()
profile["pulse_norm"] = (profile.groupby("measurement")["pulse"].transform(
    lambda s: 0.0 if len(s) <= 1 else (s - s.min()) / (s.max() - s.min())
).round(4))
```

The figure below summarizes samples with a valid blood pressure waveform and
at least one valid optical waveform (n = 31,105),

![DataProfile](/asset/DataProfile.jpg)

To apply this project to your own data, the data should be organized into 
the same three-file structure (`x.npy`, `y.npy`, and `profile.csv`).
These three files define the overall data interface assumed by the project.
Different use cases may rely on only a subset of the data or metadata fields.
For example, the dimensionality of `x.npy` (e.g., number of channels or time 
points) is flexible, and `y.npy`, which serves as labels during supervised 
training, is not required for self-supervised representation learning or 
downstream analysis. 
Please refer to the documentation and implementation of specific use cases 
for exact requirements.

## Model 

<!-- Add More Details -->

For checkpoints downloaded from [OSF](https://osf.io/yqpht/) in
[Installation](#installation), two pretrained models are provided under 
`ckpt/`:

-   `pretrain-t/epoch3885.ckpt`
    Model pretrained on unsupervised reconstruction task using optical
    waveforms, configured by
    [`config/pipeline/pretrain-t.yaml`](config/pipeline/pretrain-t.yaml).
-   `pretrain-h/last.ckpt`
    Model further pretrained with supervised regression task using optical
    waveforms and blood pressure waveforms, configured by
    [`config/pipeline/pretrain-h.yaml`](config/pipeline/pretrain-h.yaml).

The figure below illustrates backbone architecture of model.
For more details, please refer to implement in
[`src/model/model.py`](src/model/model.py).

![Model](/asset/Model.jpg)

## Evaluation

After representation learning of transformer encoders, we evaluate learned 
representations for all samples through low-dimensional visualization. 
To compute UMAP and PCA coordinates of learned representations on data
`data/raw/` (default) with given model,

```bash
python -m script.evaluation ckpt_load_path=ckpt/pretrain-t/epoch3885.ckpt
python -m script.evaluation ckpt_load_path=ckpt/pretrain-h/last.ckpt
```

If `data_save_fold` is not specified, the script assumes `ckpt_load_path` 
follows the pattern `ckpt/$name/*.ckpt` and set `data_save_fold` to
`data/evaluation/$name/` accordingly. 
Results are saved under `data_save_fold` including an updated `profile.csv`
with appended UMAP and PCA coordinates, along with copies of `x.npy` and 
`y.npy`.

Since this project uses [Hydra](https://github.com/facebookresearch/hydra)
for configuration management, additional parameters defined in
[`config/pipeline/evaluation.yaml`](config/pipeline/evaluation.yaml)
can be overridden from command line through 
[Hydra](https://github.com/facebookresearch/hydra)'s syntax.
For example, to evaluate custom data with a new model, adjust batch 
size to fit your hardware, and save results to a specific directory:

```bash
python -m script.evaluation \
    data_save_fold=path/to/your/data/save/folder/ \
    ckpt_load_path=path/to/your/checkpoint.ckpt \
    data.data_load_fold=path/to/your/data/load/folder/ \
    data.batch_size=32
```
