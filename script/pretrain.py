import hydra
import omegaconf

import lightning
import torch

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)


@hydra.main(version_base=None, config_path="../config")
def main(cfg: omegaconf.DictConfig) -> None:
    data = src.DataModule(**cfg.data)
    model = src.Model(**cfg.model)
    objective = src.ObjectivePretrain(model, **cfg.objective)
    trainer = src.Trainer(**cfg.trainer)
    trainer.fit(objective, datamodule=data)


if __name__ == "__main__": main()
