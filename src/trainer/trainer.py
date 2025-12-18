import os

import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
import torch

from ..objective import ObjectivePretrain

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self, 
        max_epochs: int,
        log_save_fold: str,
        ckpt_save_fold: str,
        every_n_epochs: int,
        ckpt_load_path: str | None = None,
        resume: bool = False,
    ) -> None:
        self.ckpt_load_path = ckpt_load_path
        self.resume = resume
        # callbacks
        logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=log_save_fold, name="", version="",
        )
        checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_save_fold, filename="epoch{epoch:04d}", 
            auto_insert_metric_name=False,
            every_n_epochs=every_n_epochs, save_top_k=-1, save_last=True,
        )
        lrmonitor = lightning.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
        # trainer
        self.trainer = lightning.Trainer(
            logger=logger, callbacks=[checkpoint, lrmonitor],
            max_epochs=max_epochs,
            log_every_n_steps=1, benchmark=True,
        )

    def fit(
        self, objective: ObjectivePretrain, 
        datamodule: lightning.LightningDataModule
    ) -> None:
        if self.ckpt_load_path is not None and not self.resume:
            Trainer.ckptLoader_(objective.model, self.ckpt_load_path)
        self.trainer.fit(
            model=objective, datamodule=datamodule,
            ckpt_path=self.ckpt_load_path if self.resume else None
        )

    @staticmethod
    def ckptFinder(
        ckpt_load_fold: str, epoch: int | None = None
    ) -> str:
        target = "last" if epoch is None else f"epoch{epoch:04d}"
        for f in os.listdir(ckpt_load_fold):
            if target in f and f.endswith(".ckpt"):
                ckpt_load_fold = os.path.join(ckpt_load_fold, f)
                return ckpt_load_fold
        raise FileNotFoundError

    @staticmethod
    def ckptLoader_(
        model: torch.nn.Module, ckpt_load_path: str
    ) -> torch.nn.Module:
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
