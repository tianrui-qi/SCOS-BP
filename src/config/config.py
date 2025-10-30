import dataclasses


__all__ = ["Config"]


class Config():
    def __init__(self):
        self.data: ConfigData = ConfigData()
        self.model: ConfigModel = ConfigModel()
        self.runner: ConfigRunner = ConfigRunner()
        self.trainer: ConfigTrainer = ConfigTrainer()


@dataclasses.dataclass(slots=True)
class ConfigData():
    x_load_path: str = "data/waveform/x.pt"
    y_load_path: str = "data/waveform/y.pt"
    split_load_path: str = "data/waveform/split.pt"
    channel_perm: bool = True
    channel_drop: bool = True
    batch_size: int = 256
    num_workers: int = 8


@dataclasses.dataclass(slots=True)
class ConfigModel():
    D: int = 256
    # tokenizer
    S: int = 40
    stride: int = 20
    # masking
    p_point: float = 0.2
    p_span_small: tuple[float, float] = (0.0, 0.5)  # channel = 1
    p_span_large: tuple[float, float] = (0.0, 1.0)  # channel > 1
    p_hide: float = 0.9
    p_keep: float = 0.1
    # embedding
    C_max: int = 8
    L_max: int = 1024
    # transformer
    num_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 1024
    # freeze
    freeze_embedding: bool = False
    freeze_transformer: int = 0


@dataclasses.dataclass(slots=True)
class ConfigRunner():
    # loss
    enable: tuple[bool, ...] = (True, True, False)
    weight: tuple[float, ...] = (0.2, 0.8, 0.0)
    T: float = 0.2
    # optimizer
    lr: float = 0.005
    step_size: int = 20
    gamma: float = 0.98


@dataclasses.dataclass(slots=True)
class ConfigTrainer():
    max_epochs: int = 10000
    log_save_fold: str = "log/"
    ckpt_save_fold: str = "ckpt/"
    monitor: str = "loss/valid"
    save_top_k: int = 10
    ckpt_load_path: str | None = None
    resume: bool = False
