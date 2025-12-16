## Set Up

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

The data and pretrained model checkpoints are available on
[OSF](https://osf.io/yqpht/). You can download them from the command line as
follows

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

## Data

For data downloaded from [OSF](https://osf.io/yqpht/) in
[Set Up](#set-up), three files are provided under `data/raw/`:

-   `x.npy`: optical waveforms (33635 samples × 3 channels × 1000 time 
    points), stored as `float32`. 
    Channels: Finger 808nm BFi, Finger 808nm PPG, Wrist 808nm BFi.
-   `y.npy`: blood pressure (BP) waveforms (33635 samples × 1000 time points), 
    stored as `float32`.
-   `profile.csv`: per-sample metadata.

The figure below illustrates the sample preparation process from raw
measurements. The bottom-right panel shows two example samples: the first
sample passes quality control for all waveforms, while the second sample
passes quality control for only two optical waveforms. Samples are retained
even when some channels are missing; in such cases, the missing channels are
represented as NaNs in `x.npy` and `y.npy`.

![DataPreparation](/asset/DataPreparation.svg)

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

![DataProfile](/asset/DataProfile.svg)

To apply this project to your own data, the data should be organized into 
the same three-file structure (`x.npy`, `y.npy`, and `profile.csv`).
These three files define the overall data interface assumed by the project.
Different use cases may rely on only a subset of the data or metadata fields.
For example, the dimensionality of `x.npy` (e.g., number of channels or time 
points) is flexible, and `y.npy`, which serves as labels during supervised 
training, is not required for self-supervised representation learning or 
downstream analysis. 
Please refer to the documentation of specific use cases for exact 
requirements.
