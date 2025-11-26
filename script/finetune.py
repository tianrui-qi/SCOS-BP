import torch
import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks

import os
import inspect
import argparse
import warnings
import dataclasses

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)
# disable MPS UserWarning: The operator 'aten::col2im' is not currently 
# supported on the MPS backend
warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")


def main() -> None:
    args = getArgs()
    for config_name in args.config_name: 
        for subject in args.subject:
            print(f"{'=' * 20}[{config_name}:{subject}]{'=' * 20}")
            train(config_name, subject)


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
    parser.add_argument(
        "--subject", "-S", type=str, nargs="+", required=True,
        dest="subject"
    )
    args = parser.parse_args()
    return args


def train(config_name: str, subject: str) -> None:
    # config
    config: src.config.Config = getattr(src.config, config_name)()
    # data
    # setup manually since we will not use datamodule=data in trainer.fit
    # we input train_dataloader and val_dataloader of trainer.fit manually
    # so that we can set subject-specific data for dataloader
    data = src.data.Module(**dataclasses.asdict(config.data))
    data.setup()
    # model
    model = src.model.SCOST(**dataclasses.asdict(config.model))
    if not config.trainer.resume and config.trainer.ckpt_load_path is not None:
        model = src.util.ckptLoader_(model, config.trainer.ckpt_load_path)
    # runner
    runner = src.runner.Finetune(model, **dataclasses.asdict(config.runner))
    # trainer
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=config.trainer.log_save_fold, 
        name=os.path.join(config_name, subject), version="",
    )
    checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(
            config.trainer.ckpt_save_fold, config_name, subject
        ),
        every_n_epochs=config.trainer.every_n_epochs,
        filename="{epoch:04d}", save_top_k=-1, save_last=True,
    )
    lrmonitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    trainer = lightning.Trainer(
        logger=logger, callbacks=[checkpoint, lrmonitor],
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=1, benchmark=True,
    )
    # fit
    trainer.fit(
        runner, train_dataloaders=data.train_dataloader(subject=subject),
        val_dataloaders=data.val_dataloader(subject=subject),
        ckpt_path=config.trainer.ckpt_load_path
        if config.trainer.resume else None
    )


if __name__ == "__main__": main()
