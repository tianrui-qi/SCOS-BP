import torch
import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks

import os
import inspect
import argparse
import dataclasses

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)


def main() -> None:
    args = getArgs()
    for config_name in args.config_name: 
        train(config_name)


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name", "-C", type=str, nargs="+", required=True, 
        dest="config_name", choices=[
            name for name, obj in inspect.getmembers(src.config) 
            if inspect.isclass(obj) 
            and issubclass(obj, src.config.Config) 
            and obj is not src.config.Config
        ],
    )
    args = parser.parse_args()
    return args


def train(config_name: str) -> None:
    # config
    config: src.config.Config = getattr(src.config, config_name)()
    # data
    data = src.data.DataModule(**dataclasses.asdict(config.data))
    # model
    model = src.model.SCOST(**dataclasses.asdict(config.model))
    if not config.trainer.resume and \
    config.trainer.ckpt_load_path is not None:
        ckpt = torch.load(
            config.trainer.ckpt_load_path, weights_only=True
        )
        state_dict = {
            k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict, strict=False)
    # runner
    runner = src.runner.Runner(
        model=model, **dataclasses.asdict(config.runner)
    )
    # trainer
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=config.trainer.log_save_fold, name=config_name, version="",
    )
    checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(config.trainer.ckpt_save_fold, config_name),
        monitor=config.trainer.monitor,
        save_top_k=config.trainer.save_top_k,
        save_last=True,
    )
    lrmonitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    trainer = lightning.Trainer(
        logger=logger,
        callbacks=[checkpoint, lrmonitor],
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=1,
        benchmark=True,
    )
    # fit
    trainer.fit(
        runner, datamodule=data,
        ckpt_path=config.trainer.ckpt_load_path
        if config.trainer.resume else None
    )


if __name__ == "__main__": main()
