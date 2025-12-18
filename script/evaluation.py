import os
import hydra
import omegaconf
import pathlib
import tqdm

import numpy as np
import sklearn.decomposition
import torch
import umap

import src


@hydra.main(
    version_base=None, config_path="../config", 
    config_name="pipeline/evaluation"
)
def main(cfg: omegaconf.DictConfig) -> None:
    # device
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    # data
    data = src.data.DataModule(**cfg.data)
    data.setup()
    # model
    model = src.model.Model(**cfg.model)
    src.trainer.Trainer.ckptLoader_(model, cfg.ckpt_load_path)
    model.to(device).eval()
    # representation
    with torch.no_grad(): representation = torch.cat([    # (N, D)
        model.forward(b[0].to(device), b[1].to(device))
        for b in tqdm.tqdm(data.test_dataloader(), desc="evaluation")
    ], dim=0).cpu().numpy()
    # umap
    u = umap.UMAP(n_jobs=1, random_state=42, verbose=True)
    u = u.fit_transform(representation)
    u = (u - u.mean(axis=0)) / u.std(axis=0)    # type: ignore
    data.profile[["umap1", "umap2"]] = u.round(4)
    # pca
    p = sklearn.decomposition.PCA(6, random_state=42)
    p = p.fit_transform(representation)
    for i in range(p.shape[1]): data.profile[f"pc{i+1}"] = p[:, i].round(4)
    # save
    # NOTE:
    # - If `cfg.data_save_fold` is not specified, the evaluation data will be
    #   auto deduced from the `cfg.ckpt_load_path`, i.e., we assume the 
    #   `ckpt_load_path` follows `ckpt/$name/*.ckpt` and set ``data_save_fold`
    #   to `data/evaluation/$name/`.
    # - We save the profile as binary `.parquet` format for efficiency and 
    #   smaller file size. In the visualization web app, we provide editor
    #   tool to view, edit, and download the profile.
    # - Visulization of waveform currently not supported. Will add later if
    #   needed.
    data_save_fold = os.path.join(
        "data/evaluation", 
        pathlib.Path(cfg.ckpt_load_path).parent.relative_to("ckpt").as_posix()
    ) if not cfg.data_save_fold else cfg.data_save_fold
    os.makedirs(data_save_fold, exist_ok=True)
    profile_save_path = os.path.join(data_save_fold, "profile.csv.parquet")
    # x_save_path = os.path.join(data_save_fold, "x.npy")
    # y_save_path = os.path.join(data_save_fold, "y.npy")
    data.profile.to_parquet(profile_save_path, index=False)
    # np.save(x_save_path, data.x.detach().cpu().numpy())
    # if cfg.data.y_as_y: np.save(y_save_path, data.y.detach().cpu().numpy())


if __name__ == "__main__": main()
