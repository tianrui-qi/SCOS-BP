import hydra
import omegaconf
import os
import pathlib
import tqdm

import lightning
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

import src


torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)


@hydra.main(
    version_base=None, config_path="../config", 
    config_name="pipeline/downstream"
)
def main(cfg: omegaconf.DictConfig):
    # device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    # data
    data = src.DataModule(**cfg.data)
    data.setup()
    # model
    model = src.Model(**cfg.model)
    model.to(device)
    # deduce data_save_fold
    data_save_fold = os.path.join(
        "data/downstream", 
        pathlib.Path(cfg.ckpt_load_path).parent.relative_to("ckpt").as_posix()
    ) if not cfg.data_save_fold else cfg.data_save_fold
    os.makedirs(data_save_fold, exist_ok=True)
    # finetune and prediction for each measurement
    z = {}  # {measurement: (N_m, T)}
    for m in tqdm.tqdm(
        data.profile[
            data.profile["split"] != 0
        ]["measurement"].unique().tolist(), 
        unit="measurement",
    ):
        # model reset
        src.trainer.Trainer.ckptLoader_(model, cfg.ckpt_load_path)
        # objective
        objective = src.ObjectiveFinetune(model, **cfg.objective)
        optimizer, scheduler = objective.configure_optimizers()
        optimizer, scheduler = optimizer[0], scheduler[0]
        # backbone outputs
        model.eval()
        x, y = [], []   # (N, L, D), (N, T), condition == 1
        for batch in data.train_dataloader(measurement=m):
            batch = [b.to(device) for b in batch]
            with torch.no_grad(): 
                x.append(model.forward(batch[0], batch[1], pool_dim=1))
            y.append(batch[2])
        x_train, y_train = torch.cat(x, dim=0), torch.cat(y, dim=0) 
        x, y = [], []   # (N, L, D), (N, T), condition != 1
        for batch in data.val_dataloader(measurement=m):
            batch = [b.to(device) for b in batch]
            with torch.no_grad(): 
                x.append(model.forward(batch[0], batch[1], pool_dim=1))
            y.append(batch[2])
        x_valid, y_valid = torch.cat(x, dim=0), torch.cat(y, dim=0)
        # finetune
        loss = []
        for epoch in tqdm.tqdm(
            range(cfg.max_epochs), desc=f"finetune: {m}", leave=False
        ):
            # train step
            model.train()
            loss_train = objective.stepAdapter((x_train, y_train))
            # backprop
            optimizer.zero_grad()
            loss_train[0].backward()
            optimizer.step()
            scheduler.step()
            # valid step
            model.eval()
            with torch.no_grad():
                loss_valid = objective.stepAdapter((x_valid, y_valid))
            loss.append(loss_valid)
        # save valid loss curve
        os.makedirs(os.path.join(data_save_fold, "log"), exist_ok=True)
        log(loss, log_save_path=os.path.join(data_save_fold, f"log/{m}.png"))
        # prediction
        model.eval()
        with torch.no_grad(): z[m] = torch.cat([     # (N_m, T)
            model.forwardRegressionAdapter(
                batch[0].to(device), batch[1].to(device)
            ).detach().cpu()
            for batch in tqdm.tqdm(
                data.test_dataloader(measurement=m), 
                desc=f"predict: {m}", leave=False
            )
        ], dim=0)
    # save
    profile_save = []
    x_save = []
    y_save = []
    z_save = []
    for m in z:
        idx = data.profile[data.profile["measurement"] == m].index.tolist()
        profile_save.append(data.profile.loc[idx])
        x_save.append(data.x[idx].numpy())
        y_save.append(data.y[idx].numpy())
        z_save.append(z[m].numpy())
    profile_save = pd.concat(profile_save, axis=0)
    x_save = np.concatenate(x_save, axis=0)
    y_save = np.concatenate(y_save, axis=0)
    z_save = np.concatenate(z_save, axis=0)
    profile_save.to_csv(
        os.path.join(data_save_fold, "profile.csv"), index=False
    )
    np.save(os.path.join(data_save_fold, "x.npy"), x_save)
    np.save(os.path.join(data_save_fold, "y.npy"), y_save)
    np.save(os.path.join(data_save_fold, "z.npy"), z_save)


def log(loss, log_save_path):
    plt.figure(figsize=(6,4))
    plt.plot([l[0].item() for l in loss], label="total")
    plt.plot([l[1].item() for l in loss], label="shape")
    plt.plot([l[2].item() for l in loss], label="min")
    plt.plot([l[3].item() for l in loss], label="max")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_save_path)
    plt.close()


if __name__ == "__main__": main()
