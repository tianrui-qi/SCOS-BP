import torch
import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks

import os
import sys
import json

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)


def main(cfg_load_path: str) -> None:
    # cfg
    with open(cfg_load_path, "r") as f: cfg = json.load(f)
    # src
    data = src.data.DataModule(**cfg["data"])
    model = src.model.SCOST(**cfg["model"])
    runner = src.runner.Pretrain(model=model, **cfg["runner"])
    # trainer
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=cfg["trainer"]["log_save_fold"],
        name=cfg_load_path.split("/")[-1].split(".")[0], 
        version="",
    )
    checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(
            cfg["trainer"]["ckpt_save_fold"], 
            cfg_load_path.split("/")[-1].split(".")[0]
        ),
        monitor=cfg["trainer"]["monitor"], 
        save_top_k=cfg["trainer"]["save_top_k"], 
        save_last=True,
    )
    lrmonitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    trainer = lightning.Trainer(
        logger=logger,
        callbacks=[checkpoint, lrmonitor],
        max_epochs=cfg["trainer"]["max_epochs"],
        log_every_n_steps=1,
    )
    # fit
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__": main(sys.argv[1])