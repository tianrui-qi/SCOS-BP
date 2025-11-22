import dataclasses

from typing import Literal


class Config():
    def __init__(self):
        self.data: ConfigData = ConfigData()
        self.model: ConfigModel = ConfigModel()
        self.runner: ConfigRunner = ConfigRunner()
        self.trainer: ConfigTrainer = ConfigTrainer()

    def eval(self):
        self.data.filter_level = "All"
        # PyTorch uses the spawn start method on macOS/Windows
        # if a script creates a DataLoader with num_workers > 0 outside
        # an if __name__ == "__main__": guard, each worker will re-import the
        # main script and recursively launch new workers, causing a crash
        self.data.num_workers = 0
        # we already close perturb and augment when using test_dataloader
        # close them again as a backup
        self.data.a_range = None
        self.data.b_range = None
        self.data.channel_perm = False
        self.data.channel_drop = 0
        self.data.channel_shift = 0
        # freeze model parameters except adapter
        self.runner.freeze_embedding = True
        self.runner.freeze_transformer = 4  # freeze all layers
        self.runner.freeze_head = True


@dataclasses.dataclass(slots=True)
class ConfigData():
    enable: tuple[bool, ...] = (True, False, True, False)
    # data
    data_load_path: str = "data/wave2wave.mat"
    y_as_channel: bool = True
    # normalize
    mu: float | None = 115.0    # set to None if want per-subject mu
    sd: float | None = 20.0     # set to None if want per-subject sd
    # split
    split_type: Literal[
        "SubjectDependent", "SubjectIndependent"
    ] | None = "SubjectIndependent"
    split_ratio: tuple[float, float, float] = (0.5, 0.2, 0.3)
    # filter
    filter_level: Literal["X", "Y", "XY", "All"] | None = None
    # dataloader
    batch_size: int = 256
    num_workers: int = 8
    # perturb
    P: int = 32
    a_range: tuple[float, float] | float | None = (0.5, 1.5)
    b_range: tuple[float, float] | float | None = (-2.0, 2.0)
    # augment
    channel_perm: bool = True
    channel_drop: float = 1     # probability of enable channel drop
    channel_shift: float = 0


@dataclasses.dataclass(slots=True)
class ConfigModel():
    D: int = 256
    # tokenizer
    S: int = 40
    stride: int = 20
    # embedding
    C_max: int = 8
    L_max: int = 1024
    # transformer
    num_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 1024


@dataclasses.dataclass(slots=True)
class ConfigRunner():
    # model
    freeze_embedding: bool = False
    freeze_transformer: int = 0
    freeze_head: bool = False   # exclude adapter
    # loss
    enable: tuple[bool, ...] = (True, False, True, False)
    weight: tuple[float, ...] = (0.2, 0.0, 0.8, 0.0)
    # contrastive
    T: float = 0.2
    # reconstruction
    p_point: float = 0.2
    p_span_small: tuple[float, float] = (0.0, 0.5)  # channel = 1
    p_span_large: tuple[float, float] = (0.0, 1.0)  # channel > 1
    p_hide: float = 0.9
    p_keep: float = 0.1
    # optimizer
    lr: float = 0.005
    step_size: int = 30
    gamma: float = 0.98


@dataclasses.dataclass(slots=True)
class ConfigTrainer():
    max_epochs: int = 10000
    log_save_fold: str = "log/"
    ckpt_save_fold: str = "ckpt/"
    every_n_epochs: int = 100
    ckpt_load_path: str | None = None
    resume: bool = False
