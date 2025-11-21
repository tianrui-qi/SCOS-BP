# %% # import
""" import """

import torch
import numpy as np
import pandas as pd
import scipy.stats

import os
import tqdm
import warnings
import dataclasses

import src

if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"

# disable MPS UserWarning: The operator 'aten::col2im' is not currently 
# supported on the MPS backend
warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")

def ckptFinder(config: src.config.Config, epoch: int | None = None) -> str:
    root = config.trainer.ckpt_save_fold
    name = config.__class__.__name__
    target = "last" if epoch is None else f"epoch={epoch}"
    for f in os.listdir(os.path.join(root, name)):
        if target in f and f.endswith(".ckpt"):
            return os.path.join(root, name, f)
    raise FileNotFoundError


# %% # config
""" config """

config = src.config.ConfigE07()     # .data and .model will be used
config.eval()
config.trainer.ckpt_load_path = ckptFinder(config, epoch=None)
print(f"load ckpt from {config.trainer.ckpt_load_path}")
# for prediction save & load
result_fold = f"data/{config.__class__.__name__}/"
result_path = os.path.join(result_fold, "result.pt")
profile_path = os.path.join(result_fold, "profile.csv")


# %% # prediction
""" prediction """

# data
dm = src.data.DataModuleReg(**dataclasses.asdict(config.data))
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
        x, channel_idx
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


# %% # calibration
""" calibration """

# calibration 0: no calibration
print("calibration 0")
for s in ["train", "test"]: print(
    s, "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
        np.nanmean(np.abs(
            profile[
                (profile["split"] == s) & (profile["condition"] != 1)
            ]["(P-T)MinBP"]
        )),
        np.nanmean(np.abs(
            profile[
                (profile["split"] == s) & (profile["condition"] != 1)
            ]["(P-T)MaxBP"]
        )),
    )
)

# calibration 1: a and b per subject and min/max
profile["Cal1PredMinBP"]  = np.nan
profile["Cal1PredMaxBP"]  = np.nan
profile["(Cal1P-T)MinBP"] = np.nan
profile["(Cal1P-T)MaxBP"] = np.nan
for subject, group in profile.groupby("subject"):
    # calibration subset: condition == 1
    cond1 = group[group["condition"] == 1]
    # fit linear models if enough calibration points
    if len(cond1) < 2:
        # Not enough calibration points: no correction
        a_min, b_min = 0.0, 0.0   # error ≈ 0 → P_adj = P
        a_max, b_max = 0.0, 0.0
    else:
        # Min BP: fit PredMinBP -> (P-T)MinBP
        a_min, b_min, r_value, p_value, std_err = scipy.stats.linregress(
            cond1["PredMinBP"], cond1["(P-T)MinBP"]
        )
        # Max BP: fit PredMaxBP -> (P-T)MaxBP
        a_max, b_max, r_value, p_value, std_err = scipy.stats.linregress(
            cond1["PredMaxBP"], cond1["(P-T)MaxBP"]
        )
    # indices of this subject in the original DataFrame
    idx = group.index
    # Apply this subject's correction to all its rows
    pred_min = profile.loc[idx, "PredMinBP"]
    pred_max = profile.loc[idx, "PredMaxBP"]
    true_min = profile.loc[idx, "TrueMinBP"]
    true_max = profile.loc[idx, "TrueMaxBP"]
    # Predicted error from linear model
    err_hat_min = a_min * pred_min + b_min  # type: ignore
    err_hat_max = a_max * pred_max + b_max  # type: ignore
    # Corrected predictions: P_adj = P - (aP + b)
    adj_pred_min = pred_min - err_hat_min
    adj_pred_max = pred_max - err_hat_max
    # stre
    profile.loc[idx, "Cal1PredMinBP"]  = adj_pred_min
    profile.loc[idx, "Cal1PredMaxBP"]  = adj_pred_max
    profile.loc[idx, "(Cal1P-T)MinBP"] = adj_pred_min - true_min
    profile.loc[idx, "(Cal1P-T)MaxBP"] = adj_pred_max - true_max
print("calibration 1")
for s in ["train", "test"]: print(
    s, "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
        np.nanmean(np.abs(
            profile[
                (profile["split"] == s) & (profile["condition"] != 1)
            ]["(Cal1P-T)MinBP"]
        )),
        np.nanmean(np.abs(
            profile[
                (profile["split"] == s) & (profile["condition"] != 1)
            ]["(Cal1P-T)MaxBP"]
        )),
    )
)

# calibration 2: a per split and min/max, b per subject and min/max
# 2.0 prepare columns for second calibration
profile["Cal2PredMinBP"]  = np.nan
profile["Cal2PredMaxBP"]  = np.nan
profile["(Cal2P-T)MinBP"] = np.nan
profile["(Cal2P-T)MaxBP"] = np.nan
# 2.1 compute global slopes for each split (train / test)
global_slopes = {}  # keys: (split, "min") / (split, "max")
for split_name in ["train", "test"]:
    df_split = profile[
        (profile["split"] == split_name) & (profile["condition"] == 1)
    ]
    # Min BP: PredMinBP -> (P-T)MinBP
    if df_split["PredMinBP"].notna().sum() >= 2:
        a_min, b_min, r, p, se = scipy.stats.linregress(
            df_split["PredMinBP"], df_split["(P-T)MinBP"]
        )
    else:
        a_min = 0.0  # no slope info; treat as pure bias
    global_slopes[(split_name, "min")] = a_min
    # Max BP: PredMaxBP -> (P-T)MaxBP
    if df_split["PredMaxBP"].notna().sum() >= 2:
        a_max, b_max, r, p, se = scipy.stats.linregress(
            df_split["PredMaxBP"], df_split["(P-T)MaxBP"]
        )
    else:
        a_max = 0.0
    global_slopes[(split_name, "max")] = a_max
# 2.2 per-subject bias estimation using fixed global slope
for subject, group in profile.groupby("subject"):
    # process each split separately, because slope depends on split
    for split_name in ["train", "test"]:
        sub = group[group["split"] == split_name]
        if sub.empty: continue  # this subject has no samples in this split
        # use global slopes
        a_min = global_slopes[(split_name, "min")]
        a_max = global_slopes[(split_name, "max")]
        # we use condition == 1 samples from this subject for bias estimation
        calib = sub[sub["condition"] == 1]
        # MinBP: bias b_min_subject
        if calib["PredMinBP"].notna().sum() >= 1:
            # (P-T)_i ≈ a_global * Pred_i + b_subject
            # => b_subject = mean((P-T)_i - a_global * Pred_i)
            b_min = np.nanmean(
                calib["(P-T)MinBP"] - a_min * calib["PredMinBP"]
            )
        else:
            b_min = 0.0  # no info, fallback to 0
        # MaxBP: bias b_max_subject
        if calib["PredMaxBP"].notna().sum() >= 1:
            b_max = np.nanmean(
                calib["(P-T)MaxBP"] - a_max * calib["PredMaxBP"]
            )
        else:
            b_max = 0.0
        # apply calibration to all rows of this subject & split
        idx = sub.index
        pred_min = profile.loc[idx, "PredMinBP"]
        pred_max = profile.loc[idx, "PredMaxBP"]
        true_min = profile.loc[idx, "TrueMinBP"]
        true_max = profile.loc[idx, "TrueMaxBP"]
        # predicted error from fixed-slope + subject-specific bias
        err_hat_min = a_min * pred_min + b_min
        err_hat_max = a_max * pred_max + b_max
        adj_pred_min = pred_min - err_hat_min
        adj_pred_max = pred_max - err_hat_max
        profile.loc[idx, "Cal2PredMinBP"]  = adj_pred_min
        profile.loc[idx, "Cal2PredMaxBP"]  = adj_pred_max
        profile.loc[idx, "(Cal2P-T)MinBP"] = adj_pred_min - true_min
        profile.loc[idx, "(Cal2P-T)MaxBP"] = adj_pred_max - true_max
# 2.3 print
print("calibration 2")
for s in ["train", "test"]: print(
    s, "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
        np.nanmean(np.abs(
            profile[
                (profile["split"] == s) & (profile["condition"] != 1)
            ]["(Cal2P-T)MinBP"]
        )),
        np.nanmean(np.abs(
            profile[
                (profile["split"] == s) & (profile["condition"] != 1)
            ]["(Cal2P-T)MaxBP"]
        )),
    )
)


# %% # visualization
""" visualization """

visualization = src.util.Visualization(result, profile)


# %%
