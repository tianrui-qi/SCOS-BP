config_name, subject, epoch = "Finetune", "S001", 1399


# %% # setup
""" setup """

import torch
import lightning
import numpy as np
import pandas as pd

import os
import tqdm
import warnings
import dataclasses

import src

torch.set_float32_matmul_precision("medium")
lightning.seed_everything(42, workers=True, verbose=False)
# disable MPS UserWarning: The operator 'aten::col2im' is not currently 
# supported on the MPS backend
warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")

# help function to get a ckpt_load_path
def ckptFinder(
    config: src.config.Config, 
    subject: str | None = None,
    epoch: int | None = None
) -> str:
    ckpt_load_path = os.path.join(
        config.trainer.ckpt_save_fold, config.__class__.__name__
    )
    if subject is not None:
        ckpt_load_path = os.path.join(ckpt_load_path, subject)
    target = "last" if epoch is None else f"epoch={epoch:04d}"
    for f in os.listdir(ckpt_load_path):
        if target in f and f.endswith(".ckpt"):
            return os.path.join(ckpt_load_path, f)
    raise FileNotFoundError

# device
if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"
# config
config: src.config.Config = getattr(src.config, config_name)().eval()
config.trainer.ckpt_load_path = ckptFinder(
    config, subject=subject, epoch=epoch
)
print(f"load ckpt from {config.trainer.ckpt_load_path}")

result_fold = f"data/{config.__class__.__name__}/"
result_path = os.path.join(result_fold, "result.pt")
profile_path = os.path.join(result_fold, "profile.csv")


# %% # prediction
""" prediction """

# data
dm = src.data.Pretrain(**dataclasses.asdict(config.data))
dm.setup()
# model
model = src.model.SCOST(**dataclasses.asdict(config.model))
ckpt = torch.load(
    config.trainer.ckpt_load_path, weights_only=True, 
    map_location=torch.device(device)
)
state_dict = {
    k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
    if k.startswith("model.")
}
model.load_state_dict(state_dict, strict=False)
model = model.eval().to(device)
# predict
result_b = []
for batch in tqdm.tqdm(dm.test_dataloader()):
    # batch to device
    x, channel_idx, y = batch
    x, channel_idx, y = x.to(device), channel_idx.to(device), y.to(device)
    # forward
    with torch.no_grad(): x_pred = model.forwardRegression(
        x, channel_idx, adapter=True
    )
    # store result
    result_b.append(torch.cat([
        x.detach().cpu(),                       # (B, 4, T)
        y.detach().cpu().unsqueeze(1),          # (B, 1, T)
        x_pred.detach().cpu().unsqueeze(1),     # (B, 1, T)
    ], dim=1))
result = torch.cat(result_b, dim=0)            # (N, 5, T)
# map bp in 3nd and 4th channel to waveform before normalization
# and store new waveform in 5th and 6th channel
result = torch.cat([
    result, 
    dm.denormalize(result[:, 3, :]).unsqueeze(1),
    dm.denormalize(result[:, 4, :]).unsqueeze(1),
], dim=1).detach().cpu()
# store key features in profile
profile = dm.profile
profile["split"] = profile["split"].map({0: "train", 1: "test", 2: "test"})
profile["system"] = profile["system"].map({False: "old", True: "new"})
profile["TrueMinBP"] = result[:, 5].min(dim=1).values.numpy()
profile["TrueMaxBP"] = result[:, 5].max(dim=1).values.numpy()
profile["PredMinBP"] = result[:, 6].min(dim=1).values.numpy()
profile["PredMaxBP"] = result[:, 6].max(dim=1).values.numpy()
profile["(P-T)MinBP"] = profile["PredMinBP"] - profile["TrueMinBP"]
profile["(P-T)MaxBP"] = profile["PredMaxBP"] - profile["TrueMaxBP"]
# save result as .pt and sample as .csv in result_save_fold
# print shape and where saved
os.makedirs(result_fold, exist_ok=True)
torch.save(result, result_path)
profile.to_csv(profile_path, index=False)
print(f"sample: pd.DataFrame > {profile_path}\t{profile.shape}")
print(f"result: torch.Tensor > {result_path}\t\t{tuple(result.shape)}")


# %% # load
""" load """

result = torch.load(result_path, weights_only=True)
profile = pd.read_csv(profile_path)

print("train", "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
    np.nanmean(np.abs(
        profile[
            (profile["subject"] == subject) & (profile["condition"] == 1)
        ]["(P-T)MinBP"]
    )),
    np.nanmean(np.abs(
        profile[
            (profile["subject"] == subject) & (profile["condition"] == 1)
        ]["(P-T)MaxBP"]
    )),
))
print("valid", "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
    np.nanmean(np.abs(
        profile[
            (profile["subject"] == subject) & (profile["condition"] != 1)
        ]["(P-T)MinBP"]
    )),
    np.nanmean(np.abs(
        profile[
            (profile["subject"] == subject) & (profile["condition"] != 1)
        ]["(P-T)MaxBP"]
    )),
))


# %% # visualization
""" visualization """

visualization = src.util.Visualization(result, profile)


# %%
