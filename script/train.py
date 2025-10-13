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


def main(config_load_path: str) -> None:
    # config
    with open(config_load_path, "r") as f: config = json.load(f)
    # src
    data = src.data.DataModule(**config["data"])
    model = src.model.SCOST(**config["model"])
    runner = src.runner.Pretrain(model=model, **config["runner"])
    # trainer
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=config["trainer"]["log_save_fold"],
        name=config_load_path.split("/")[-1].split(".")[0], 
        version="",
    )
    checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(
            config["trainer"]["ckpt_save_fold"], 
            config_load_path.split("/")[-1].split(".")[0]
        ),
        monitor=config["trainer"]["monitor"], 
        save_top_k=config["trainer"]["save_top_k"], 
        save_last=True,
    )
    lrmonitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    trainer = lightning.Trainer(
        logger=logger,
        callbacks=[checkpoint, lrmonitor],
        max_epochs=config["trainer"]["max_epochs"],
        log_every_n_steps=1,
    )
    # fit
    trainer.fit(runner, datamodule=data)


if __name__ == "__main__": main(sys.argv[1])