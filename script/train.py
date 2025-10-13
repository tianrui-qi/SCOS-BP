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
    # data
    data = src.data.DataModule(**config["data"])
    # model
    model = getattr(src.model, config["model"]["class"])
    model: torch.nn.Module = model(**config["model"])
    if not config["trainer"]["resume"] and \
    config["trainer"]["ckpt_load_path"] is not None:
        ckpt = torch.load(
            config["trainer"]["ckpt_load_path"], weights_only=True
        )
        state_dict = {
            k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict, strict=False)
    # runner
    runner = getattr(src.runner, config["runner"].pop("class"))
    runner: lightning.LightningModule = runner(model=model, **config["runner"])
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
        benchmark=True,
    )
    # fit
    trainer.fit(
        runner, datamodule=data,
        ckpt_path=config["trainer"]["ckpt_load_path"] 
        if config["trainer"]["resume"] else None
    )


if __name__ == "__main__": main(sys.argv[1])