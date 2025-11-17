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
import plotly.express
import plotly.subplots
import plotly.graph_objects
import ipywidgets
from IPython.display import display, clear_output

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

config = src.config.ConfigE01()     # .data and .model will be used
config.trainer.ckpt_load_path = ckptFinder(config, epoch=None)
print(f"load ckpt from {config.trainer.ckpt_load_path}")
# for prediction save & load
result_fold = f"data/{config.__class__.__name__}/"
result_path = os.path.join(result_fold, "result.pt")
sample_path = os.path.join(result_fold, "sample.csv")


# %% # prediction
""" prediction """

# load data
x = torch.load(
    os.path.join(config.data.data_load_fold, 'x.pt'), weights_only=True
)
# normalize each channel
mean = torch.nanmean(x, dim=(0, 2))
mean2 = torch.nanmean(x * x, dim=(0, 2))
std = torch.sqrt(mean2 - mean * mean)
x = (x - mean.view(1, -1, 1)) / std.view(1, -1, 1)
# sample profile
sample = pd.read_csv(os.path.join(config.data.data_load_fold, "sample.csv"))
sample = sample.drop(columns=["split02"])
sample = sample.rename(columns={"split01": "split"})
sample["split"] = sample["split"].map({0: "train", 1: "test", 2: "test"})
sample["system"] = sample["system"].map({False: "old", True: "new"})
# filter valid samples
valid = torch.where(~torch.isnan(x).any(dim=(1, 2)))[0]
x = x[valid]
sample = sample.iloc[valid.numpy()].reset_index(drop=True)
# data
dataloader = torch.utils.data.DataLoader(
    src.data.RawDataset(x), 
    batch_size=config.data.batch_size, shuffle=False, 
)
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
for batch in tqdm.tqdm(dataloader):
    # batch to device
    x, channel_idx = batch
    x, channel_idx = x.to(device), channel_idx.to(device)
    # forward
    with torch.no_grad(): x_pred, _ = model.forwardReconstructionRaw(
        x, channel_idx, user_mask=3
    )
    # store result
    result_b.append(torch.cat([
        x.detach().cpu(),                               # (B, 4, T)
        x_pred[:, 3, :].detach().cpu().unsqueeze(1)     # (B, 1, T)
    ], dim=1))
result = torch.cat(result_b, dim=0)            # (N, 5, T)
# map bp in 3nd and 4th channel to waveform before normalization
# and store new waveform in 5th and 6th channel
result = torch.cat([
    result, 
    (result[:, 3, :] * std[3] + mean[3]).unsqueeze(1),
    (result[:, 4, :] * std[3] + mean[3]).unsqueeze(1),
], dim=1).detach().cpu()
# store key features in profile
sample["TrueMinBP"] = result[:, 5].min(dim=1).values.numpy()
sample["TrueMaxBP"] = result[:, 5].max(dim=1).values.numpy()
sample["PredMinBP"] = result[:, 6].min(dim=1).values.numpy()
sample["PredMaxBP"] = result[:, 6].max(dim=1).values.numpy()
sample["(P-T)MinBP"] = sample["PredMinBP"] - sample["TrueMinBP"]
sample["(P-T)MaxBP"] = sample["PredMaxBP"] - sample["TrueMaxBP"]
# save result as .pt and sample as .csv in result_save_fold
# print shape and where saved
os.makedirs(result_fold, exist_ok=True)
torch.save(result, result_path)
sample.to_csv(sample_path, index=False)
print(f"sample: pd.DataFrame > {sample_path}\t{sample.shape}")
print(f"result: torch.Tensor > {result_path}\t\t{tuple(result.shape)}")


# %% # load
""" load """

result = torch.load(result_path, weights_only=True)
sample = pd.read_csv(sample_path)


# %% # calibration
""" calibration """

# region # calibration 0: no calibration
print("calibration 0")
for s in ["train", "test"]: print(
    s, "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
        np.nanmean(np.abs(
            sample[
                (sample["split"] == s) & (sample["condition"] != 1)
            ]["(P-T)MinBP"]
        )),
        np.nanmean(np.abs(
            sample[
                (sample["split"] == s) & (sample["condition"] != 1)
            ]["(P-T)MaxBP"]
        )),
    )
)
# endregion

# region # calibration 1: a and b per subject and min/max
sample["Cal1PredMinBP"]  = np.nan
sample["Cal1PredMaxBP"]  = np.nan
sample["(Cal1P-T)MinBP"] = np.nan
sample["(Cal1P-T)MaxBP"] = np.nan
for subject, group in sample.groupby("subject"):
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
    pred_min = sample.loc[idx, "PredMinBP"]
    pred_max = sample.loc[idx, "PredMaxBP"]
    true_min = sample.loc[idx, "TrueMinBP"]
    true_max = sample.loc[idx, "TrueMaxBP"]
    # Predicted error from linear model
    err_hat_min = a_min * pred_min + b_min  # type: ignore
    err_hat_max = a_max * pred_max + b_max  # type: ignore
    # Corrected predictions: P_adj = P - (aP + b)
    adj_pred_min = pred_min - err_hat_min
    adj_pred_max = pred_max - err_hat_max
    # stre
    sample.loc[idx, "Cal1PredMinBP"]  = adj_pred_min
    sample.loc[idx, "Cal1PredMaxBP"]  = adj_pred_max
    sample.loc[idx, "(Cal1P-T)MinBP"] = adj_pred_min - true_min
    sample.loc[idx, "(Cal1P-T)MaxBP"] = adj_pred_max - true_max
print("calibration 1")
for s in ["train", "test"]: print(
    s, "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
        np.nanmean(np.abs(
            sample[
                (sample["split"] == s) & (sample["condition"] != 1)
            ]["(Cal1P-T)MinBP"]
        )),
        np.nanmean(np.abs(
            sample[
                (sample["split"] == s) & (sample["condition"] != 1)
            ]["(Cal1P-T)MaxBP"]
        )),
    )
)
# endregion

# region # calibration 2: a per split and min/max, b per subject and min/max
# 2.0 prepare columns for second calibration
sample["Cal2PredMinBP"]  = np.nan
sample["Cal2PredMaxBP"]  = np.nan
sample["(Cal2P-T)MinBP"] = np.nan
sample["(Cal2P-T)MaxBP"] = np.nan
# 2.1 compute global slopes for each split (train / test)
global_slopes = {}  # keys: (split, "min") / (split, "max")
for split_name in ["train", "test"]:
    df_split = sample[
        (sample["split"] == split_name) & (sample["condition"] == 1)
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
for subject, group in sample.groupby("subject"):
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
        pred_min = sample.loc[idx, "PredMinBP"]
        pred_max = sample.loc[idx, "PredMaxBP"]
        true_min = sample.loc[idx, "TrueMinBP"]
        true_max = sample.loc[idx, "TrueMaxBP"]
        # predicted error from fixed-slope + subject-specific bias
        err_hat_min = a_min * pred_min + b_min
        err_hat_max = a_max * pred_max + b_max
        adj_pred_min = pred_min - err_hat_min
        adj_pred_max = pred_max - err_hat_max
        sample.loc[idx, "Cal2PredMinBP"]  = adj_pred_min
        sample.loc[idx, "Cal2PredMaxBP"]  = adj_pred_max
        sample.loc[idx, "(Cal2P-T)MinBP"] = adj_pred_min - true_min
        sample.loc[idx, "(Cal2P-T)MaxBP"] = adj_pred_max - true_max
# 2.3 print
print("calibration 2")
for s in ["train", "test"]: print(
    s, "\tMAE of (min, max) = ({:5.2f}, {:5.2f})".format(
        np.nanmean(np.abs(
            sample[
                (sample["split"] == s) & (sample["condition"] != 1)
            ]["(Cal2P-T)MinBP"]
        )),
        np.nanmean(np.abs(
            sample[
                (sample["split"] == s) & (sample["condition"] != 1)
            ]["(Cal2P-T)MaxBP"]
        )),
    )
)
# endregion


# %% # visualization
""" visualization """

# ========== Global constants / config ==========
COMMON_FONT = 12
BOOL_STR_MAP = {
    False: "false", True: "true",
    "False": "false", "True": "true",
    0: "false", 1: "true"
}
COLOR_SEQ = plotly.express.colors.qualitative.Plotly
SYMBOL_SEQ = ["circle", "square", "diamond", "x", "cross", "triangle-up"]

# Columns used for filtering / hover
filter_cols = ["subject", "health", "system", "repeat", "condition", "split",]
hover_cols = ["subject", "health", "system", "repeat", "condition", "split",]
multi_filter_cols = ["subject", "condition"]
bool_like_cols   = ["health", "system", "repeat", "split"]

def compute_waveform_ranges(waveforms: np.ndarray):
    """
    Compute global y-range for the two waveform rows.

    Args:
        waveforms: 
            array of shape (N, 7, T) containing waveforms for all samples.

    Returns:
        wave_range_row1: 
            [ymin, ymax] for row 1 (channels 5-6, "before normalization")
        wave_range_row2: 
            [ymin, ymax] for row 2 (channels 0-4, "after normalization")
        N_T: number of time points
        GLOBAL_T: shared time axis (0..T-1)
    """
    # Top row: ch5–6 (BP before normalization)
    wave_min_row1 = float(waveforms[:, 5:7, :].min())
    wave_max_row1 = float(waveforms[:, 5:7, :].max())

    # Bottom row: ch0–4 (after normalization)
    wave_min_row2 = float(waveforms[:, 0:5, :].min())
    wave_max_row2 = float(waveforms[:, 0:5, :].max())

    # Add small padding so curves don't sit exactly at the border
    pad1 = 0.05 * (wave_max_row1 - wave_min_row1 + 1e-8)
    pad2 = 0.05 * (wave_max_row2 - wave_min_row2 + 1e-8)

    wave_range_row1 = [wave_min_row1 - pad1, wave_max_row1 + pad1]
    wave_range_row2 = [wave_min_row2 - pad2, wave_max_row2 + pad2]

    N_T = waveforms.shape[2]
    GLOBAL_T = np.arange(N_T)

    return wave_range_row1, wave_range_row2, N_T, GLOBAL_T

def create_controls(sample: pd.DataFrame, numeric_cols: list[str]):
    """
    Create all interactive widgets (controls) for the dashboard.

    Returns a dict containing:
        - x1_dropdown, y1_dropdown, x2_dropdown, y2_dropdown
        - color_dropdown, size_input, opacity_input
        - multi_filter_widgets, bool_filter_radios
        - widgets_list: flat list of widgets to bind update callbacks
        - left_col: pre-assembled left column layout for UI
    """
    UNIFIED_WIDTH = "220px"
    CONTROL_STYLE = {"description_width": "70px"}

    # ---- axes selection (x1, y1, x2, y2) ----
    x1_dropdown = ipywidgets.Dropdown(
        options=numeric_cols, value="PredMinBP", description="x1",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE,
    )
    y1_dropdown = ipywidgets.Dropdown(
        options=numeric_cols, value="(P-T)MinBP", description="y1",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE,
    )
    x2_dropdown = ipywidgets.Dropdown(
        options=[None] + numeric_cols, value=None, description="x2",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE,
    )
    y2_dropdown = ipywidgets.Dropdown(
        options=[None] + numeric_cols, value=None, description="y2",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE,
    )

    # ---- color by categorical column ----
    color_dropdown = ipywidgets.Dropdown(
        options=filter_cols, value="condition", description="color",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE,
    )

    # ---- point size and opacity ----
    size_input = ipywidgets.BoundedIntText(
        value=4, min=1, max=15, step=1, description="size",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE,
    )
    opacity_input = ipywidgets.BoundedFloatText(
        value=0.7, min=0.1, max=1.0, step=0.05, description="opacity",
        layout=ipywidgets.Layout(width=UNIFIED_WIDTH), style=CONTROL_STYLE
    )

    # ---- multi-value filters (subject / condition) using SelectMultiple ----
    multi_filter_widgets = {}
    for col in multi_filter_cols:
        unique_vals = sorted(sample[col].unique())
        options = [(str(v), v) for v in unique_vals]
        sm = ipywidgets.SelectMultiple(
            options=options,
            value=tuple(v for _, v in options),  # select all by default
            description=col,
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH, height="125px"),
            style=CONTROL_STYLE,
        )
        multi_filter_widgets[col] = sm

    # ---- boolean-like filters using Dropdown ----
    bool_filter_radios = {}
    for col in bool_like_cols:
        unique_vals = sorted(sample[col].unique())
        # For some columns we show lowercase Boolean labels
        options = (
            [("all", "__ALL__")] + 
            [(str(v).lower(), v) for v in unique_vals]
        ) if col in ["health", "repeat"] else (
            [("all", "__ALL__")] + [(str(v), v) for v in unique_vals]
        )
        radio = ipywidgets.Dropdown(
            options=options,
            value="test" if col == "split" else "__ALL__",
            description=col,
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )
        bool_filter_radios[col] = radio

    # ========= Left-side UI layout (all controls in one HBox) =========
    left_col = ipywidgets.HBox([
        ipywidgets.VBox([
            x1_dropdown, y1_dropdown,
            x2_dropdown, y2_dropdown,
        ]),
        ipywidgets.VBox([
            color_dropdown,
            size_input,
            opacity_input,
        ]),
        multi_filter_widgets["subject"],
        multi_filter_widgets["condition"],
        ipywidgets.VBox([
            bool_filter_radios["health"],
            bool_filter_radios["system"],
            bool_filter_radios["repeat"],
            bool_filter_radios["split"],
        ]),
    ])

    # ========= Widgets list to attach the update callback =========
    widgets_list = [
        x1_dropdown, y1_dropdown,
        x2_dropdown, y2_dropdown,
        color_dropdown,
    ] + list(
        multi_filter_widgets.values()
    ) + list(bool_filter_radios.values()) + [size_input, opacity_input]

    return {
        "x1_dropdown": x1_dropdown,
        "y1_dropdown": y1_dropdown,
        "x2_dropdown": x2_dropdown,
        "y2_dropdown": y2_dropdown,
        "color_dropdown": color_dropdown,
        "size_input": size_input,
        "opacity_input": opacity_input,
        "multi_filter_widgets": multi_filter_widgets,
        "bool_filter_radios": bool_filter_radios,
        "widgets_list": widgets_list,
        "left_col": left_col,
    }

def build_dashboard(sample: pd.DataFrame, result: torch.Tensor):
    """
    Build and display the full scatter + waveform dashboard.

    Args:
        sample: dataframe with metadata and numeric prediction/metric columns.
        result: torch.Tensor of shape (N, 7, T) holding waveforms to visualize.

    Returns:
        The top-level ipywidgets container (VBox).
    """
    # ------- Prepare waveform data and global ranges -------
    waveforms = result.detach().cpu().numpy()  # (N, 7, 1000)
    wave_range_row1, wave_range_row2, N_T, GLOBAL_T = (
        compute_waveform_ranges(waveforms)
    )

    # Numeric columns for x/y axes selection
    numeric_cols = sample.select_dtypes(include=["float64"]).columns.tolist()
    # Create all control widgets
    controls = create_controls(sample, numeric_cols)
    plot_output = ipywidgets.Output()

    # Global state for update_plot (captured via nonlocal)
    # prevents recursive updates when we modify widget options
    suppress_update = False
    # position of "sample_idx" inside customdata array
    sample_idx_pos = None
    # mapping from semantic name to trace index for waveform       
    wave_trace_indices = {}

    def update_plot(*args):
        """
        Main callback to:
          1) apply filters based on current widget values
          2) build scatter + waveform subplots
          3) wire up click callbacks to show waveform for a selected point
        """
        nonlocal suppress_update, sample_idx_pos, wave_trace_indices
        if suppress_update: return

        with plot_output:
            clear_output(wait=True)

            # Step 1: Apply boolean-like filters 
            df_bool = sample
            for col, w in controls["bool_filter_radios"].items():
                val = w.value
                if val == "__ALL__":
                    continue
                df_bool = df_bool[df_bool[col] == val]
            if df_bool.empty:
                # If no samples left, clear multi filters and bail out
                suppress_update = True
                try:
                    for col, w in controls["multi_filter_widgets"].items():
                        w.options = []
                        w.value = ()
                finally:
                    suppress_update = False
                print("No samples match the current filters.")
                return

            # Step 2: Update options for subject / condition based on df_bool 
            suppress_update = True
            try:
                for col, w in controls["multi_filter_widgets"].items():
                    new_vals = sorted(df_bool[col].unique())
                    new_options = [(str(v), v) for v in new_vals]

                    old_selected = list(w.value)
                    new_selected = (
                        tuple(v for v in old_selected if v in new_vals)
                    )

                    # If nothing from old selection is valid, fallback to all
                    if len(new_selected) == 0:
                        new_selected = tuple(new_vals)

                    w.options = new_options
                    w.value = new_selected
            finally:
                suppress_update = False

            # Step 3: Apply multi-value filters (subject / condition) 
            sub_df = df_bool
            for col, w in controls["multi_filter_widgets"].items():
                selected = list(w.value)
                if len(selected) > 0: 
                    sub_df = sub_df[sub_df[col].isin(selected)]

            if sub_df.empty:
                print("No samples match the current filters.")
                return

            # Add "sample_idx" representing row index in original `sample`
            sub_df = sub_df.copy()
            sub_df["sample_idx"] = sub_df.index

            # Step 4: Prepare data for scatter (pair1 / pair2) 
            x1 = controls["x1_dropdown"].value
            y1 = controls["y1_dropdown"].value
            x2 = controls["x2_dropdown"].value
            y2 = controls["y2_dropdown"].value
            has_x2 = (x2 is not None) and (y2 is not None)

            color_col = controls["color_dropdown"].value
            point_size = controls["size_input"].value
            point_opacity = controls["opacity_input"].value

            df_list = []

            # Pair 1
            df1 = sub_df.copy()
            df1["_x"] = df1[x1]
            df1["_y"] = df1[y1]
            df1["_pair"] = "pair1"
            df_list.append(df1)

            # Optional pair 2
            if has_x2:
                df2 = sub_df.copy()
                df2["_x"] = df2[x2]
                df2["_y"] = df2[y2]
                df2["_pair"] = "pair2"
                df_list.append(df2)

            df_long = pd.concat(df_list, ignore_index=True)

            # --- Color mapping for scatter ---
            if color_col:
                df_long["_color_label"] = (
                    df_long[color_col]
                    .map(BOOL_STR_MAP)
                    .fillna(df_long[color_col])
                    .astype(str)
                ) if color_col in ["health", "repeat"] else (
                    df_long[color_col].astype(str)
                )
                color_vals = sorted(df_long["_color_label"].unique())
            else:
                df_long["_color_label"] = "all"
                color_vals = ["all"]
            color_map = {
                v: COLOR_SEQ[i % len(COLOR_SEQ)] 
                for i, v in enumerate(color_vals)
            }

            # --- Marker shape mapping for x1 vs y1 / x2 vs y2 ---
            pair_vals = sorted(df_long["_pair"].unique())
            symbol_map = {p: SYMBOL_SEQ[i] for i, p in enumerate(pair_vals)}
            if "pair1" not in symbol_map: symbol_map["pair1"] = SYMBOL_SEQ[0]
            if "pair2" not in symbol_map: symbol_map["pair2"] = SYMBOL_SEQ[1]

            pair_label_map = {
                "pair1": "x1 vs y1",
                "pair2": "x2 vs y2",
            }

            # ====== Build hovertemplate and customdata for scatter ======
            sample_idx_series = df_long["sample_idx"].astype(int)
            sample_idx_arr = sample_idx_series.to_numpy()

            sub_indexed = sub_df.set_index("sample_idx")

            # Real x1/y1 values (not the _x/_y used for pair stacking)
            x1_for_row = sub_indexed.loc[sample_idx_arr, x1].to_numpy()
            y1_for_row = sub_indexed.loc[sample_idx_arr, y1].to_numpy()

            custom_columns = []
            lines = []

            # First entries in customdata: x1, y1
            custom_columns.append(x1_for_row)
            custom_columns.append(y1_for_row)
            lines.append("x1 = %{customdata[0]}")
            lines.append("y1 = %{customdata[1]}")
            next_idx = 2

            # x2, y2 if present
            if has_x2:
                x2_for_row = sub_indexed.loc[sample_idx_arr, x2].to_numpy()
                y2_for_row = sub_indexed.loc[sample_idx_arr, y2].to_numpy()
                custom_columns.append(x2_for_row)
                custom_columns.append(y2_for_row)
                lines.append(f"x2 = %{{customdata[{next_idx}]}}")
                lines.append(f"y2 = %{{customdata[{next_idx+1}]}}")
                next_idx += 2

            # sample index "s"
            sample_idx_pos = next_idx
            custom_columns.append(sample_idx_arr)
            lines.append(f"s = %{{customdata[{next_idx}]}}")
            next_idx += 1

            # metadata columns in hover
            for col in hover_cols:
                if col in ["health", "repeat"]:
                    vals = (
                        df_long[col]
                        .map(BOOL_STR_MAP)
                        .fillna(df_long[col])
                        .astype(str)
                        .to_numpy()
                    )
                else:
                    vals = df_long[col].astype(str).to_numpy()
                custom_columns.append(vals)
                lines.append(f"{col} = %{{customdata[{next_idx}]}}")
                next_idx += 1

            hover_template = "<br>".join(lines) + "<extra></extra>"
            customdata = np.column_stack(custom_columns)

            # Build 2×2 subplot layout: 
            # left scatter, right-top/right-bottom waveform
            base_fig = plotly.subplots.make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"rowspan": 2}, {"type": "xy"}],
                    [None, {"type": "xy"}]
                ],
                column_widths=[0.5, 0.5],
                row_heights=[0.5, 0.5],
                horizontal_spacing=0.035,
                vertical_spacing=0.035,
                # share x-axis for right-top and right-bottom waveform plots
                shared_xaxes=True,   
            )
            fig = plotly.graph_objects.FigureWidget(base_fig)

            # --- Scatter traces (main data) ---
            if color_col:
                for c in color_vals:
                    for p in pair_vals:
                        mask = (
                            (df_long["_color_label"] == c) & 
                            (df_long["_pair"] == p)
                        )
                        if not mask.any():
                            continue
                        sub = df_long[mask]
                        fig.add_trace(
                            plotly.graph_objects.Scatter(
                                x=sub["_x"],
                                y=sub["_y"],
                                mode="markers",
                                marker=dict(
                                    color=color_map[c],
                                    symbol=symbol_map.get(p, SYMBOL_SEQ[0]),
                                    size=point_size,
                                    opacity=point_opacity,
                                ),
                                customdata=customdata[mask.to_numpy()],
                                hovertemplate=hover_template,
                                showlegend=False,
                            ),
                            row=1, col=1,
                        )
            else:
                # No color grouping: use a single color
                for p in pair_vals:
                    mask = (df_long["_pair"] == p)
                    if not mask.any():
                        continue
                    sub = df_long[mask]
                    fig.add_trace(
                        plotly.graph_objects.Scatter(
                            x=sub["_x"],
                            y=sub["_y"],
                            mode="markers",
                            marker=dict(
                                color="blue",
                                symbol=symbol_map.get(p, SYMBOL_SEQ[0]),
                                size=point_size,
                                opacity=point_opacity,
                            ),
                            customdata=customdata[mask.to_numpy()],
                            hovertemplate=hover_template,
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

            # ========= Placeholder waveform traces =========
            t = GLOBAL_T
            color_bp_true = "#1f77b4"
            color_bp_pred = "#ff7f0e"

            waveform_hover = (
                "T = %{x}<br>"
                "BP True = %{customdata[0]:.2f}<br>"
                "BP Pred = %{customdata[1]:.2f}<br>"
                "|Pred-True| = %{customdata[2]:.2f}<extra></extra>"
            )

            # Placeholder customdata for waveform (will be filled on click)
            empty_wave_custom = np.zeros((N_T, 3), dtype=float)

            # Row 1 right: BP True / BP Pred before normalization
            start_idx = len(fig.data)
            idx_bp_before_true = start_idx
            fig.add_trace(
                plotly.graph_objects.Scatter(
                    x=t,
                    y=np.full(N_T, np.nan),
                    mode="lines",
                    name="BP True",
                    line=dict(color=color_bp_true, width=2),
                    legendgroup="waveform",
                    legendgrouptitle_text="waveform",
                    showlegend=True,
                    customdata=empty_wave_custom,
                    hovertemplate=waveform_hover,
                ),
                row=1, col=2,
            )
            idx_bp_before_pred = start_idx + 1
            fig.add_trace(
                plotly.graph_objects.Scatter(
                    x=t,
                    y=np.full(N_T, np.nan),
                    mode="lines",
                    name="BP Pred",
                    line=dict(color=color_bp_pred, width=2),
                    legendgroup="waveform",
                    showlegend=True,
                    customdata=empty_wave_custom,
                    hovertemplate=waveform_hover,
                ),
                row=1, col=2,
            )

            # Row 2 right: ch0-2 + BP True/Pred after normalization
            idx_ch0 = len(fig.data)
            for ch in range(3):
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=t,
                        y=np.full(N_T, np.nan),
                        mode="lines",
                        name=f"ch{ch}",
                        opacity=0.5,
                        line=dict(width=2.0),
                        legendgroup="waveform",
                        showlegend=True,
                    ),
                    row=2, col=2,
                )
            idx_bp_after_true = idx_ch0 + 3
            fig.add_trace(
                plotly.graph_objects.Scatter(
                    x=t,
                    y=np.full(N_T, np.nan),
                    mode="lines",
                    name="BP True",
                    line=dict(color=color_bp_true, width=2),
                    legendgroup="waveform",
                    showlegend=False,
                ),
                row=2, col=2,
            )
            idx_bp_after_pred = idx_ch0 + 4
            fig.add_trace(
                plotly.graph_objects.Scatter(
                    x=t,
                    y=np.full(N_T, np.nan),
                    mode="lines",
                    name="BP Pred",
                    line=dict(color=color_bp_pred, width=2),
                    legendgroup="waveform",
                    showlegend=False,
                ),
                row=2, col=2,
            )

            # Map human-readable names to trace indices for later updates
            wave_trace_indices = {
                "bp_before_true": idx_bp_before_true,
                "bp_before_pred": idx_bp_before_pred,
                "ch0": idx_ch0,
                "ch1": idx_ch0 + 1,
                "ch2": idx_ch0 + 2,
                "bp_after_true": idx_bp_after_true,
                "bp_after_pred": idx_bp_after_pred,
            }

            # Scatter legend: first shapes (pair1 vs pair2), then colors
            # Shape legend entries (x1 vs y1 / x2 vs y2)
            for i, p in enumerate(["pair1", "pair2"]):
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(
                            color="black",
                            size=COMMON_FONT,
                            symbol=symbol_map.get(p, SYMBOL_SEQ[i]),
                        ),
                        name=pair_label_map[p],
                        showlegend=True,
                        legendgroup="scatter",
                        legendgrouptitle_text="scatter" if i == 0 else None,
                    )
                )

            # Color legend entries
            for c in color_vals:
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(
                            color=color_map[c],
                            size=COMMON_FONT,
                        ),
                        name=f"{color_col} = {c}",
                        showlegend=True,
                        legendgroup="scatter",
                    )
                )

            # ========= Axes ranges & layout =========
            fig.update_yaxes(row=1, col=2, range=wave_range_row1)
            fig.update_yaxes(row=2, col=2, range=wave_range_row2)

            fig.update_xaxes(
                showticklabels=False,
                row=1, col=2,
            )
            fig.update_xaxes(row=2, col=2)

            fig.update_layout(
                height=520,
                width=1200,
                font=dict(size=COMMON_FONT),
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="v"),
            )

            # Click callback: update waveform by clicked scatter point
            def handle_click(trace, points, state_click):
                """
                On clicking a scatter point, look up the corresponding
                waveform index and update all waveform traces accordingly.
                """
                if not points.point_inds: return
                idx0 = points.point_inds[0]
                cd_row = trace.customdata[idx0]
                s_idx = int(cd_row[sample_idx_pos])
                if (s_idx < 0) or (s_idx >= waveforms.shape[0]): return

                w = waveforms[s_idx]
                # Expect (7, T): 5 channels + 2 BP waveforms
                if (w.ndim != 2) or (w.shape[0] < 7): return

                bp_true = w[5]
                bp_pred = w[6]
                bp_diff = np.abs(bp_true - bp_pred)

                # Global MAE for min/max, shown in annotation
                TrueMaxBP = float(bp_true.max())
                PredMaxBP = float(bp_pred.max())
                TrueMinBP = float(bp_true.min())
                PredMinBP = float(bp_pred.min())
                mae_max = abs(TrueMaxBP - PredMaxBP)
                mae_min = abs(TrueMinBP - PredMinBP)

                # Per-timepoint customdata for waveform hover: 
                # [true, pred, |pred-true|]
                wave_custom = np.stack([bp_true, bp_pred, bp_diff], axis=1)

                with fig.batch_update():
                    # Update before-normalization traces
                    fig.data[wave_trace_indices["bp_before_true"]].y = bp_true
                    fig.data[wave_trace_indices["bp_before_pred"]].y = bp_pred

                    fig.data[
                        wave_trace_indices["bp_before_true"]
                    ].customdata = wave_custom
                    fig.data[
                        wave_trace_indices["bp_before_pred"]
                    ].customdata = wave_custom

                    # Update after-normalization traces: ch0-2 + BP True/Pred
                    fig.data[wave_trace_indices["ch0"]].y = w[0]
                    fig.data[wave_trace_indices["ch1"]].y = w[1]
                    fig.data[wave_trace_indices["ch2"]].y = w[2]
                    fig.data[wave_trace_indices["bp_after_true"]].y = w[3]
                    fig.data[wave_trace_indices["bp_after_pred"]].y = w[4]

                    # Annotation in the top-right waveform subplot
                    fig.update_layout(annotations=[dict(
                        xref="x2 domain",
                        yref="y2 domain",
                        x=1.0,
                        y=0.0,
                        text=(
                            "MAE of (min, max) = " +
                            f"({mae_min:.2f}, {mae_max:.2f})"
                        ),
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.7)",
                        borderwidth=1,
                        font=dict(size=COMMON_FONT),
                    )])

            # Attach click handler only to scatter traces with customdata
            for tr in fig.data:
                if (
                    getattr(tr, "mode", None) == "markers"
                    and getattr(tr, "customdata", None) is not None
                ): tr.on_click(handle_click)

            display(fig)

    # ========= Bind events to update_plot =========
    for w in controls["widgets_list"]: w.observe(update_plot, names="value")

    # Top-level UI: controls on top, plot below
    ui = ipywidgets.VBox([controls["left_col"], plot_output])
    display(ui)
    update_plot()
    return ui

ui = build_dashboard(sample, result)


# %%
