import torch

import os

from ..config import Config
from ..model import SCOST


def ckptFinder(
    config: Config, subject: str | None = None, epoch: int | None = None
) -> str:
    ckpt_load_path = os.path.join(
        config.trainer.ckpt_save_fold, config.__class__.__name__
    )
    if subject is not None: 
        ckpt_load_path = os.path.join(ckpt_load_path, subject)
    target = "last" if epoch is None else f"epoch={epoch:04d}"
    for f in os.listdir(ckpt_load_path):
        if target in f and f.endswith(".ckpt"):
            ckpt_load_path = os.path.join(ckpt_load_path, f)
            return ckpt_load_path
    raise FileNotFoundError


def ckptLoader_(model: SCOST, ckpt_load_path: str) -> SCOST:
    ckpt = torch.load(
        ckpt_load_path, weights_only=True, 
        map_location=torch.device("cpu")
    )
    state_dict = {
        k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=False)
    return model
