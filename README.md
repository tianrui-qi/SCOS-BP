Please check the project 
[presentation](https://cdn.jsdelivr.net/gh/tianrui-qi/SCOS-BP@main/asset/presentation.pdf)
for a quick overview, the 
[manuscript](https://cdn.jsdelivr.net/gh/tianrui-qi/SCOS-BP@main/asset/manuscript.pdf)
for a detailed description, and the web app at 
[scos-bp.streamlit.app](https://scos-bp.streamlit.app)
for interactive visualization of results 
(may take a few seconds to load on first visit).

## Installation

### Environment

This project is built using
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) 
(`lightning=2.5`) for deep learning model development, 
[Hydra](https://github.com/facebookresearch/hydra) (`hydra-core=1.3`) for 
configuration management, and [Plotly](https://plotly.com/) together with
[Streamlit](https://streamlit.io/) for interactive visualization.
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

### Data and Pretrained Models

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

### Sanity Check

We provide a sanity-check script to verify environment and data are correctly 
set up. 
This script trains the model on a single fixed batch for few steps using the 
reconstruction objective. 
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

Three files are provided on [OSF](https://osf.io/yqpht/) under `data/raw/`,

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

![fig-DataPreparation](/asset/readme/fig-DataPreparation.jpg)

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

![fig-DataProfile](/asset/readme/fig-DataProfile.jpg)

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

The figure below illustrates backbone architecture of model.
For more details, please refer to implement in
[`src/model/model.py`](src/model/model.py).

![fig-ModelArchitecture](/asset/readme/fig-ModelArchitecture.jpg)

Two pretrained models checkpoints are provided on [OSF](https://osf.io/yqpht/)
under `ckpt/`,

-   `pretrain-t/epoch3885.ckpt`
    Model pretrained on unsupervised reconstruction task using optical
    waveforms. (stage 1)
-   `pretrain-h/last.ckpt`
    Model further pretrained with supervised regression task using optical
    waveforms and blood pressure waveforms. (stage 2)

The figure below illustrates the complete three-stage training pipeline.
Note that checkpoints for stage 3 are not provided, as this stage performs
measurement-specific finetuning, where a separate model is trained for each
measurement. This finetuning step is computationally lightweight and 
typically completes within minutes. Please refer to
[Finetune and Prediction](#finetune-and-prediction) section for details.

![fig-ModelTraining](/asset/readme/fig-ModelTraining.jpg)

## Pretrain

To reproduce the pretraining of provided models,

```bash
# pretrain configured by `config/pipeline/pretrain-t.yaml`
python -m script.pretrain +pipeline=pretrain-t
# pretrain configured by `config/pipeline/pretrain-h.yaml`
python -m script.pretrain +pipeline=pretrain-h
```

We use [Hydra](https://github.com/facebookresearch/hydra)'s syntax to 
define, manage, and override configuration parameters.
All pretraining settings are defined declaratively in `.yaml` files under
[`config/`](config/) and can be modified directly from the command line.
For example, to reuse an existing configuration but change the batch size:

```bash
# pretrain configured by `config/pipeline/pretrain-t.yaml`
python -m script.pretrain +pipeline=pretrain-t data.batch_size=32
```

To define a new experiment, create a new `.yaml` file under
[`config/`](config/), for example `config/custom/experiment/01.yaml`,

```yaml
# @package _global_

defaults:
  - /schema/data@_here_
  - /schema/model@_here_
  - /schema/objective@_here_
  - /schema/trainer@_here_
  - _self_

name: experiment/01

data:
  batch_size: 32
```

and launch pretraining with

```bash
python -m script.pretrain +custom=experiment/01
```

[Hydra](https://github.com/facebookresearch/hydra) also supports running 
multiple experiments with parameter combinations via 
[`hydra.mode=MULTIRUN`](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/).
As an example, [`config/experiment/b/14.yaml`](config/experiment/b/14.yaml)
defines a multi-run over `data.batch_size`.
Please refer to [Hydra](https://github.com/facebookresearch/hydra)'s
[documentation](https://hydra.cc/docs/intro/)
for additional configuration features.

Note that training log and model checkpoints are automatically saved under
`log/$name/` and `ckpt/$name/` respectively, where `$name` is defined in the
configuration file. Remember to set different names for different experiments
to avoid overwriting previous results. To check training log,

```bash
tensorboard --logdir log/
```

We highly modularized the [pretraining pipeline](script/pretrain.py) into 
four components: data, model, objective, and trainer.
We strictly followed [PyTorch](https://github.com/pytorch/pytorch) and
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) API
in our implementation. More specifically,

```
src/
├── data/
│   ├── datamodule.py   # inherits: lightning.LightningDataModule
│   └── dataset.py      # inherits: torch.utils.data.Dataset
├── model/
│   └── model.py        # inherits: torch.nn.Module
├── objective/
│   └── pretrain.py     # inherits: lightning.LightningModule
└── trainer/
    └── trainer.py      # wrapper:  lightning.Trainer
```

Thus, current pipeline can be easily modify and extended by following the 
API. 
Please check the implementation for more details.

## Evaluation

After representation learning, we compute representations for all samples and
project them into a low-dimensional space using UMAP and PCA for 
visualization. 
We developed an web app at 
[scos-bp.streamlit.app](https://scos-bp.streamlit.app) 
using [Plotly](https://plotly.com/) for plotting,
[Streamlit](https://streamlit.io/) as the frontend framework, and
[Streamlit Community Cloud](https://streamlit.io/cloud) for deployment.
You can also run the app locally by

```bash
streamlit run website/app.py
```

Results for two pretrained models are provided under 
[`data/evaluation/`](data/evaluation/).
By default, the web app will load results from 
[`pretrain-t/profile.csv.parquet`](data/evaluation/pretrain-t/profile.csv.parquet)
for demonstration.
To explore other results, simply upload a `.csv` or `.parquet` file through 
`dataframe` tab in the web app interface.

If you wish to run the [evaluation pipeline](script/evaluation.py) yourself 
on provided data and pretrained models,

```bash
python -m script.evaluation ckpt_load_path=ckpt/pretrain-t/epoch3885.ckpt
python -m script.evaluation ckpt_load_path=ckpt/pretrain-h/last.ckpt
```

If `data_save_fold` is not specified, the script assumes 
`ckpt_load_path` follows the pattern `ckpt/$name/*.ckpt` and set 
`data_save_fold` to `data/evaluation/$name/` accordingly. 
Results are saved under `data_save_fold` including 
-   `profile.csv` (for readability) and `profile.csv.parquet` (for 
    visualization), an updated profile with appended UMAP/PCA coordinates.
-   `r.npy` containing representations of samples.
-   `x.npy` and `y.npy` with the same filtering
    rules (controlled by `data.filter_level`) applied during evaluation
    so that all outputs remain aligned in length and order.

Additional parameters defined in
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

## Finetune and Prediction

To perform measurement-specific finetuning and prediction using a pretrained
model,

```bash
python -m script.downstream ckpt_load_path=ckpt/pretrain-h/last.ckpt
```

If `data_save_fold` is not specified, the script assumes 
`ckpt_load_path` follows the pattern `ckpt/$name/*.ckpt` and set 
`data_save_fold` to `data/downstream/$name/` accordingly. 
Results are saved under `data_save_fold`, including 

-   `z.npy` containing predicted blood pressure waveform. 
-   `profile.csv`, `x.npy`, and `y.npy` with the same filtering rules 
    (controlled by `data.filter_level`) applied during finetuning and 
    prediction so that all outputs remain aligned in length and order.

Additional parameters defined in
[`config/pipeline/downstream.yaml`](config/pipeline/downstream.yaml)
can be overridden from command line through 
[Hydra](https://github.com/facebookresearch/hydra)'s syntax.

This implementation serves as a reference for running the finetuning and
prediction pipeline.
Further hyperparameter tuning is required for optimal performance.

## Acknowledgements

This project was developed by 
[Tianrui Qi](https://www.linkedin.com/in/tianrui-qi/) during his Ph.D. lab 
rotation in [Biomedical Optical Technologies Lab](https://www.bu.edu/botlab/)
at Boston University.
Thanks [Dr. Darren Roblyer](https://www.linkedin.com/in/roblyer/) for hosting
the rotation, and
[Dr. Ariane Garrett](https://www.linkedin.com/in/ariane-garrett-800363157/)
and [Ana Perez](https://www.linkedin.com/in/ana-perez-b7a297207/)
for their support throughout the project.

## References

1.  Garrett, A. *et al.* Speckle contrast optical spectroscopy for cuffless 
    blood pressure estimation based on microvascular blood flow and volume 
    oscillations. *Biomedical Optics Express* **16**, 3004–3016 (2025).
    doi:[10.1364/BOE.560022](https://doi.org/10.1364/BOE.560022)

2.  Yang, C., Westover, M. B. & Sun, J. BIOT: Cross-data biosignal learning
    in the wild (2023). arXiv:[2305.10351](https://arxiv.org/abs/2305.10351)

3.  Wang, Y., Li, T., Yan, Y., Song, W. & Zhang, X. How to evaluate your 
    medical time series classification? (2024).
    arXiv: [2410.03057](https://arxiv.org/pdf/2410.03057)
