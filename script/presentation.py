# %% # setup
""" setup """

import os
import logging
import hydra

import lightning
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import src

lightning.seed_everything(42, workers=True, verbose=False)
# disable matplotlib findfont warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# config
with hydra.initialize(version_base=None, config_path="config"):
    cfg = hydra.compose(config_name="pipeline/pretrain-h")


# %% # Data: Preparation
""" Data: Preparation """

def saveSubject(
    x: torch.Tensor,          # (N, 3, 1000)
    y: torch.Tensor,          # (N, 1000)  -> BP
    num_samples: int = 10,
    line_width: float = 1.0,
    out_dir: str = "wave_svgs",
):
    """
    Save each channel and each sample as an individual SVG file.

    - Channels: ch0, ch1, ch2 (optical) and BP (blood pressure)
    - For each channel, take the first `num_samples` samples
    - Each figure contains a single 1D waveform
    - The y-axis range of each figure is based on that waveform's own min/max
    - If a waveform is entirely NaN/Inf, it is skipped

    NOTE: this function were generated with ChatGPT.
    """
    os.makedirs(out_dir, exist_ok=True)

    N, C, T = x.shape
    assert C == 3, f"Expect 3 optical channels, got {C}"
    assert y.shape == (N, T), f"y shape {y.shape}, expected {(N, T)}"

    k = min(num_samples, N)

    # Merge into 4 channels: 3 optical + 1 BP
    y_expanded = y.unsqueeze(1)                 # (N, 1, T)
    xy_all = torch.cat([x, y_expanded], dim=1)  # (N, 4, T)

    xy_sel = xy_all[:k]                         # (k, 4, T)
    xy_np = xy_sel.detach().cpu().numpy()       # (k, 4, T)

    channel_names  = ["ch0", "ch1", "ch2", "BP"]
    channel_colors = ["C0",  "C1",  "C2",  "C3"]    # default colors

    saved_count = 0

    for ch in range(4):
        color = channel_colors[ch]
        name = channel_names[ch]
        for i in range(k):
            wave = xy_np[i, ch]   # (T,)
            t = np.arange(T)

            # Consider only finite values
            finite_mask = np.isfinite(wave)
            if not finite_mask.any():
                # Entire waveform is NaN/Inf; skip
                continue

            wave_valid = wave[finite_mask]
            wave_min = float(wave_valid.min())
            wave_max = float(wave_valid.max())

            # Add small padding around min/max for nicer visualization
            span = wave_max - wave_min
            pad = span * 0.05 if span > 0 else 1.0
            y_lo = wave_min - pad
            y_hi = wave_max + pad

            plt.figure(figsize=(6, 2))
            plt.plot(t, wave, linewidth=line_width, color=color)
            plt.ylim(y_lo, y_hi)

            # Remove ticks and frame for clean export into Affinity
            plt.xticks([])
            plt.yticks([])
            plt.box(False)

            filename = os.path.join(out_dir, f"sample{i:02d}_{name}.svg")
            plt.savefig(
                filename,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()
            saved_count += 1

    print(
        f"saved: {out_dir}/sample{{i}}_{{ch}}.svg ({saved_count} files)"
    )

    # Return full (N, 4, T) tensor for later combined plots
    return xy_all

def saveSample(
    xy_all: torch.Tensor,      # (N, 4, T)
    sample_idx: int = 4,
    line_width: float = 2.0,
    out_dir: str = "wave_svgs",
    draw_ch0: bool = True,
    draw_ch1: bool = True,
    draw_ch2: bool = True,
    draw_bp: bool = True,
):
    """
    Draw a single figure with two stacked subplots:

    Top subplot:
    -   Optical channels (ch0, ch1, ch2) plotted together
    -   Which channels are drawn is controlled by draw_ch0/draw_ch1/draw_ch2
    -   Only the left y-axis spine is shown (no x-axis ticks/labels/line)

    Bottom subplot:
    -   BP channel (index 3) alone
    -   Drawn if draw_bp is True
    -   Left y + bottom x axes shown
    -   x-axis has tick labels (numbers) but no tick marks 
    (no little tick lines)

    Both subplots share the same x-axis (time).

    NOTE: this function were generated with ChatGPT.
    """
    os.makedirs(out_dir, exist_ok=True)

    xy_np_all = xy_all.detach().cpu().numpy()  # (N, 4, T)
    N, C4, T = xy_np_all.shape
    assert C4 == 4, f"Expect 4 channels (3 optical + 1 BP), got {C4}"
    assert 0 <= sample_idx < N, f"sample_idx out of range: {sample_idx}"

    t = np.arange(T)
    channel_colors = ["C0", "C1", "C2", "C3"]

    fig, (ax_opt, ax_bp) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 4)
    )

    # ===== Top: stacked optical channels =====
    if draw_ch0:
        wave = xy_np_all[sample_idx, 0]
        ax_opt.plot(
            t,
            wave,
            linewidth=line_width,
            color=channel_colors[0],
        )
    if draw_ch1:
        wave = xy_np_all[sample_idx, 1]
        ax_opt.plot(
            t,
            wave,
            linewidth=line_width,
            color=channel_colors[1],
        )
    if draw_ch2:
        wave = xy_np_all[sample_idx, 2]
        ax_opt.plot(
            t,
            wave,
            linewidth=line_width,
            color=channel_colors[2],
        )

    # Fixed y-limits for optical (as you tuned before)
    ax_opt.set_ylim(-3, 5)

    # Remove all x-axis visuals for the top plot
    ax_opt.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )

    # Only keep the left y-axis spine
    ax_opt.spines["left"].set_visible(True)
    ax_opt.spines["right"].set_visible(False)
    ax_opt.spines["top"].set_visible(False)
    ax_opt.spines["bottom"].set_visible(False)

    # ===== Bottom: BP only =====
    if draw_bp:
        bp_wave = xy_np_all[sample_idx, 3]
        ax_bp.plot(
            t,
            bp_wave,
            linewidth=line_width,
            color=channel_colors[3],
        )

    # Fixed y-limits for BP (as you tuned before)
    ax_bp.set_ylim(80, 145)

    # Keep left y + bottom x axes
    ax_bp.spines["left"].set_visible(True)
    ax_bp.spines["bottom"].set_visible(True)
    ax_bp.spines["right"].set_visible(False)
    ax_bp.spines["top"].set_visible(False)

    # Remove x tick marks but keep numeric labels
    ax_bp.tick_params(
        axis="x",
        which="both",
        bottom=False,      # no small tick lines
        top=False,
        labelbottom=True,  # but keep tick labels (numbers)
    )

    # Normal y-axis ticks/labels on the left
    ax_bp.tick_params(axis="y", which="both", left=True, labelleft=True)

    plt.tight_layout()
    filename = os.path.join(out_dir, f"sample{sample_idx:02d}.svg")
    fig.savefig(
        filename,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    print(f"saved: {filename}")

# data
data = src.DataModule(**cfg.data)
data.setup()
data.denormalize_()

xy_all = saveSubject(
    data.x, data.y,
    num_samples=20, line_width=2.0, 
    out_dir="data/presentation/DataPreparation",
)
saveSample(
    xy_all, 
    sample_idx=4, line_width=2.0, 
    out_dir="data/presentation/DataPreparation",
)
saveSample(
    xy_all,
    sample_idx=9, line_width=2.0, 
    out_dir="data/presentation/DataPreparation",
    draw_ch2=False, draw_bp=False,
)


# %% # Data: Profile
""" Data: Profile """

# data
data = src.DataModule(**cfg.data)
data.setup()

# ============================================================
# 1. Per-sample availability
# ============================================================

x = data.x          # (N, 3, T)
y = data.y          # (N, T)
profile = data.profile.copy()

N, C, T = x.shape
assert C == 3, f"Expect 3 optical channels, got {C}"
assert y.shape == (N, T), f"y shape mismatch: {y.shape}, expected {(N, T)}"

# Per-sample availability
optical_available = ~torch.isnan(x).all(dim=-1)   # (N, 3) per channel
bp_available      = ~torch.isnan(y).all(dim=-1)   # (N,)

num_optical = optical_available.sum(dim=-1).cpu().numpy()
bp_av_np    = bp_available.cpu().numpy()

# "usable": at least one optical channel + BP available
usable   = (num_optical > 0) & bp_av_np

# "complete": all 3 optical channels + BP available
complete = (num_optical == 3) & bp_av_np

profile = profile.reset_index(drop=True)
profile["usable"]   = usable
profile["complete"] = complete

# ============================================================
# 2. Collapse subject -> person, split first vs repeat
#    e.g. H001, H001_R -> person = H001, is_repeat True/False
# ============================================================

profile["is_repeat"] = profile["measurement"].str.endswith("_R")

persons    = sorted(profile["subject"].unique())
conditions = sorted(profile["condition"].unique())

num_person = len(persons)
num_cond   = len(conditions)

# Matrices: shape = (num_person, num_cond)
usable_first    = np.zeros((num_person, num_cond), dtype=int)
complete_first  = np.zeros((num_person, num_cond), dtype=int)
usable_repeat   = np.zeros((num_person, num_cond), dtype=int)
complete_repeat = np.zeros((num_person, num_cond), dtype=int)

for i, p in enumerate(persons):
    mask_person   = (profile["subject"] == p)
    mask_first    = mask_person & (~profile["is_repeat"])
    mask_repeat   = mask_person & ( profile["is_repeat"])

    for j, cond in enumerate(conditions):
        m_first_cond  = mask_first  & (profile["condition"] == cond)
        m_repeat_cond = mask_repeat & (profile["condition"] == cond)

        usable_first[i, j]    = profile.loc[
            m_first_cond,  "usable"].sum()      # type: ignore
        complete_first[i, j]  = profile.loc[
            m_first_cond,  "complete"].sum()    # type: ignore
        usable_repeat[i, j]   = profile.loc[
            m_repeat_cond, "usable"].sum()      # type: ignore
        complete_repeat[i, j] = profile.loc[
            m_repeat_cond, "complete"].sum()    # type: ignore

# ============================================================
# 3. Global heights and y-scale (for left panels)
# ============================================================

first_heights  = usable_first.sum(axis=1)
repeat_heights = usable_repeat.sum(axis=1)

max_first  = first_heights.max()   # max person total for first
max_repeat = repeat_heights.max()  # max person total for repeat
ymax       = max(max_first, max_repeat)
global_ylim = (0, ymax)

# Relative subplot heights (top vs bottom) by their maxima
total_max = max_first + max_repeat
h1 = max_first  / total_max
h2 = max_repeat / total_max

# ============================================================
# 4. Global totals over ALL persons & BOTH measurements
#    (for right-side single summary bar)
# ============================================================

# sum over persons, then over first+repeat
total_usable_cond   = (usable_first + usable_repeat).sum(axis=0)
total_complete_cond = (complete_first + complete_repeat).sum(axis=0)

total_usable_all   = total_usable_cond.sum()
total_complete_all = total_complete_cond.sum()
right_ymax = total_usable_all  # y-axis for summary bar

# ============================================================
# 5. Matplotlib font settings (Helvetica) & shared style
# ============================================================

plt.rcParams["font.family"]      = "Helvetica"
plt.rcParams["font.size"]        = 12
plt.rcParams["xtick.labelsize"]  = 12
plt.rcParams["ytick.labelsize"]  = 12
plt.rcParams["axes.labelsize"]   = 12
plt.rcParams["axes.titlesize"]   = 12
plt.rcParams["legend.fontsize"]  = 12

linewidth = 0.6

colors = plt.colormaps["Accent"].colors     # type: ignore
x_pos = np.arange(num_person)
bar_width = 0.7  # unify bar thickness for left & right

# ============================================================
# 6. Create figure with 2x2 grid:
#    left column: top (1st m.) / bottom (2nd m.)
#    right column: single axis spanning both rows (summary)
# ============================================================

fig = plt.figure(figsize=(18.4, 8.2), dpi=100)
gs  = gridspec.GridSpec(
    2,
    2,
    width_ratios=[num_person, 1],
    height_ratios=[h1, h2],
    figure=fig,
)

ax_top    = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_right  = fig.add_subplot(gs[:, 1])  # spans both rows

# ============================================================
# 7. TOP PANEL: 1st measurement, per person
# ============================================================

bottom = np.zeros(num_person)

for j in range(num_cond):
    # usable: colored solid
    ax_top.bar(
        x_pos,
        usable_first[:, j],
        bottom=bottom,
        color=colors[j],
        alpha=0.85,
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )
    # complete: hatched overlay
    ax_top.bar(
        x_pos,
        complete_first[:, j],
        bottom=bottom,
        facecolor="none",
        hatch="////",
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )
    bottom += usable_first[:, j]

ax_top.set_ylim(global_ylim)
ax_top.grid(axis="y", alpha=0.6)

# spines: keep left, remove top/right/bottom
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(True)
ax_top.spines["left"].set_visible(True)

ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
ax_top.set_xlim(-0.5, num_person - 0.5)

# label total samples above each bar (1st m.)
for i, total_count in enumerate(first_heights):
    if total_count > 0:
        ax_top.text(
            x_pos[i],
            total_count + ymax * 0.01,
            f"{total_count}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

# ============================================================
# 8. BOTTOM PANEL: 2nd measurement (repeat), per person
# ============================================================

bottom = np.zeros(num_person)

for j in range(num_cond):
    ax_bottom.bar(
        x_pos,
        usable_repeat[:, j],
        bottom=bottom,
        color=colors[j],
        alpha=0.85,
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )
    ax_bottom.bar(
        x_pos,
        complete_repeat[:, j],
        bottom=bottom,
        facecolor="none",
        hatch="////",
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )
    bottom += usable_repeat[:, j]

ax_bottom.set_ylim(global_ylim)
ax_bottom.grid(axis="y", alpha=0.6)

ax_bottom.spines["top"].set_visible(False)
ax_bottom.spines["right"].set_visible(False)
ax_bottom.spines["left"].set_visible(True)
ax_bottom.spines["bottom"].set_visible(True)

ax_bottom.set_xticks(x_pos)
ax_bottom.set_xticklabels(persons, rotation=60, ha="right")

# label total samples above each bar (2nd m.)
for i, total_count in enumerate(repeat_heights):
    if total_count > 0:
        ax_bottom.text(
            x_pos[i],
            total_count + ymax * 0.01,
            f"{total_count}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

# ============================================================
# 9. RIGHT PANEL: simplified summary bar (usable + complete)
#    - No condition colors
#    - One bar at x=0
#    - Solid outline = usable, hatched top = complete
#    - Show numeric labels for usable and complete
#    - Column width matches left side bars
# ============================================================

x_bar = np.array([0])

# aggregate totals across all persons and both measurements
total_usable_all   = usable_first.sum() + usable_repeat.sum()
total_complete_all = complete_first.sum() + complete_repeat.sum()

# --- Usable main bar (outline only, no fill) ---
ax_right.bar(
    x_bar[0],
    total_usable_all,
    color="none",
    edgecolor="black",
    width=bar_width,        # SAME WIDTH AS LEFT PANEL
    linewidth=linewidth,
)

# --- Complete overlay (hatched top) ---
ax_right.bar(
    x_bar[0],
    total_complete_all,
    bottom=0,
    facecolor="none",
    hatch="////",
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
)

# === Add extra y-ticks at usable & complete totals ===
ax_right.set_yticks([total_complete_all, total_usable_all])
ax_right.margins(y=0)

# --- Spine styles ---
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.spines["bottom"].set_visible(True)

# --- X ticks: a single label ---
ax_right.set_xticks(x_bar)
ax_right.set_xticklabels(["All"], fontfamily="Helvetica")

ax_right.tick_params(
    axis="y", which="both", labelright=True, labelleft=False, rotation=-60
)

plt.tight_layout()
print("saved: data/presentation/DataProfile.png")
os.makedirs("data/presentation", exist_ok=True)
plt.savefig(
    "data/presentation/DataProfile.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
plt.close(fig)


# %% # Model: Backbone Architecture
""" Model: Backbone Architecture """

def save_tokens_and_grid(
    token: torch.Tensor,           # (C, L, S)
    out_dir: str = "token_svgs",
    line_width: float = 1.5,
    per_token_figsize=(2.0, 2.0),  # size of each small cube
):
    """
    token: shape (C, L, S)
        C = #channels (e.g., 3 optical + 1 BP)
        L = #tokens per channel (e.g., 37)
        S = length of each token (e.g., 100)

    This function will:
        Save each token as an individual small SVG ("cube") for each 
        channel.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- convert to numpy and get shapes ----
    token_np = token.detach().cpu().numpy()   # (C, L, S)
    C, L, S = token_np.shape

    # compute y-range per channel so all tokens of that channel share scale
    # shape (C, L, S) -> (C, L*S)
    token_flat = token_np.reshape(C, -1)
    ch_mins = token_flat.min(axis=1)
    ch_maxs = token_flat.max(axis=1)

    # small padding
    pad = 0.05
    y_lims = []
    for c in range(C):
        vmin, vmax = ch_mins[c], ch_maxs[c]
        span = vmax - vmin
        if span <= 0:
            span = 1.0
        y_lims.append((vmin - pad * span, vmax + pad * span))

    # channel names: if 4 channels, treat the last one as BP
    if C == 4:
        channel_names = ["ch0", "ch1", "ch2", "BP"]
    else:
        channel_names = [f"ch{c}" for c in range(C)]

    # per-channel colors (matplotlib default cycle)
    channel_colors = [f"C{c}" for c in range(C)]

    for c in range(C):
        y_lo, y_hi = y_lims[c]
        color = channel_colors[c % len(channel_colors)]
        for l in range(L):
            wave = token_np[c, l]         # (S,)
            t = np.arange(S)

            plt.figure(figsize=per_token_figsize)
            plt.plot(t, wave, linewidth=line_width, color=color)

            # unify y scale for this channel
            plt.ylim(y_lo, y_hi)

            # remove ticks / frame for clean "cube"
            plt.xticks([])
            plt.yticks([])
            plt.box(False)

            fname = os.path.join(
                out_dir,
                f"{channel_names[c]}_token{l:02d}.svg"
            )
            plt.savefig(
                fname,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()

    print(f"saved: {out_dir}/{{ch}}_token{{l}}.svg ({C*L} files)")

# data
data = src.DataModule(**cfg.data)
data.setup()
# model
model = src.model.Model(**cfg.model)

sample_idx = 4
# original waveform
x_sample = data.x[sample_idx]            # (3, 1000), optical channels
y_sample = data.y[sample_idx].unsqueeze(0)  # (1, 1000), BP channel
# stack into 4 channels: [ch0, ch1, ch2, BP]
sample_4ch = torch.cat([x_sample, y_sample], dim=0)  # (4, 1000)
# tokenized waveform for 4 channels
token = model.tokenizer.forward(    # (4, L, S)
    sample_4ch.unsqueeze(0)
).squeeze(0)

save_tokens_and_grid(
    token,
    out_dir="data/presentation/Model",
    line_width=12,
    per_token_figsize=(1.8, 1.8),
)


# %% # Data: Stage 1 & 2 Split
""" Data: Stage 1 & 2 Split """

# ============================================================
# 1. Per-sample availability
# ============================================================

x = data.x          # (N, 3, T)
y = data.y          # (N, T)
profile = data.profile.copy()

N, C, T = x.shape
assert C == 3, f"Expect 3 optical channels, got {C}"
assert y.shape == (N, T), f"y shape mismatch: {y.shape}, expected {(N, T)}"

# Per-sample availability
optical_available = ~torch.isnan(x).all(dim=-1)   # (N, 3) per channel
bp_available      = ~torch.isnan(y).all(dim=-1)   # (N,)

num_optical = optical_available.sum(dim=-1).cpu().numpy()
bp_av_np    = bp_available.cpu().numpy()

# "usable": at least one optical channel + BP available
usable   = (num_optical > 0) & bp_av_np

# "complete": all 3 optical channels + BP available
complete = (num_optical == 3) & bp_av_np

profile = profile.reset_index(drop=True)
profile["usable"]   = usable
profile["complete"] = complete

# ============================================================
# 2. Collapse subject -> person, split first vs repeat
#    e.g. H001, H001_R -> person = H001, is_repeat True/False
# ============================================================

profile["is_repeat"] = profile["measurement"].str.endswith("_R")

persons    = sorted(profile["subject"].unique())
conditions = sorted(profile["condition"].unique())

num_person = len(persons)
num_cond   = len(conditions)

# Matrices: shape = (num_person, num_cond)
usable_first    = np.zeros((num_person, num_cond), dtype=int)
complete_first  = np.zeros((num_person, num_cond), dtype=int)
usable_repeat   = np.zeros((num_person, num_cond), dtype=int)
complete_repeat = np.zeros((num_person, num_cond), dtype=int)

for i, p in enumerate(persons):
    mask_person   = (profile["subject"] == p)
    mask_first    = mask_person & (~profile["is_repeat"])
    mask_repeat   = mask_person & ( profile["is_repeat"])

    for j, cond in enumerate(conditions):
        m_first_cond  = mask_first  & (profile["condition"] == cond)
        m_repeat_cond = mask_repeat & (profile["condition"] == cond)

        usable_first[i, j]    = profile.loc[
            m_first_cond,  "usable"].sum()      # type: ignore
        complete_first[i, j]  = profile.loc[
            m_first_cond,  "complete"].sum()    # type: ignore
        usable_repeat[i, j]   = profile.loc[
            m_repeat_cond, "usable"].sum()      # type: ignore
        complete_repeat[i, j] = profile.loc[
            m_repeat_cond, "complete"].sum()    # type: ignore

# ============================================================
# 2.5 Per-measurement split (train/test), separately for first/repeat
#      split = 0 -> train
#      split = 1 or 2 -> test
# ============================================================

# For each person, define the measurement ids
first_meas  = np.array([p for p in persons])          # e.g. H001
repeat_meas = np.array([f"{p}_R" for p in persons])   # e.g. H001_R

def meas_split(m_id: str) -> int:
    s = profile.loc[profile["measurement"] == m_id, "split"].unique()
    return int(s[0]) if len(s) else -1   # -1 if missing

split_first  = np.array([meas_split(m) for m in first_meas], dtype=int)
split_repeat = np.array([meas_split(m) for m in repeat_meas], dtype=int)

is_train_first_person  = (split_first == 0)
is_test_first_person   = np.isin(split_first,  [1, 2])

is_train_repeat_person = (split_repeat == 0)
is_test_repeat_person  = np.isin(split_repeat, [1, 2])

# train / test color
train_color = "#1f77b4"
test_color  = "#ff7f0e"

# ============================================================
# 3. Global heights and y-scale (for left panels)
# ============================================================

first_heights  = usable_first.sum(axis=1)
repeat_heights = usable_repeat.sum(axis=1)

max_first  = first_heights.max()   # max person total for first
max_repeat = repeat_heights.max()  # max person total for repeat
ymax       = max(max_first, max_repeat)
global_ylim = (0, ymax)

# Relative subplot heights (top vs bottom) by their maxima
total_max = max_first + max_repeat
h1 = max_first  / total_max
h2 = max_repeat / total_max

# ============================================================
# 4. Global totals over ALL persons & BOTH measurements
#    (for right-side single summary bar)
# ============================================================

# sum over persons, then over first+repeat
total_usable_cond   = (usable_first + usable_repeat).sum(axis=0)
total_complete_cond = (complete_first + complete_repeat).sum(axis=0)

total_usable_all   = total_usable_cond.sum()
total_complete_all = total_complete_cond.sum()
right_ymax = total_usable_all  # y-axis for summary bar

# ============================================================
# 4.5 Stage-1 usage: shown vs not-shown, and train / test / unused
#    shown part: samples actually used in Stage 1
#    We assume:
#      - train: use "usable" (>=1 optical + BP)
#      - test : use "complete" (all 3 optical + BP)
# ============================================================

split_arr = profile["split"].to_numpy()
is_train_sample = (split_arr == 0)
is_test_sample  = np.isin(split_arr, [1, 2])
is_other_sample = ~(is_train_sample | is_test_sample)

# Samples actually used in Stage 1:
#   train: usable & train
#   test : complete & test
shown_mask = (
    (usable & is_train_sample) |
    (complete & is_test_sample)
)
shown_mask = shown_mask.astype(bool)
not_shown_mask = ~shown_mask

shown_total = int(shown_mask.sum())    # expected 21592
N_total     = int(len(profile))        # expected 31105

# --- Lower part (shown): train / test / unused ---
shown_train  = int((shown_mask & is_train_sample).sum())
shown_test   = int((shown_mask & is_test_sample).sum())
shown_other  = int((shown_mask & is_other_sample).sum())

# --- Upper part (not-shown): train / test / unused ---
notshown_train = int((not_shown_mask & is_train_sample).sum())
notshown_test  = int((not_shown_mask & is_test_sample).sum())
notshown_other = int((not_shown_mask & is_other_sample).sum())

assert (
    shown_train + shown_test + shown_other +
    notshown_train + notshown_test + notshown_other
) == N_total, "counts do not sum to total N"

# ============================================================
# 5. Matplotlib font settings (Helvetica) & shared style
# ============================================================

plt.rcParams["font.family"]      = "Helvetica"
plt.rcParams["font.size"]        = 12
plt.rcParams["xtick.labelsize"]  = 12
plt.rcParams["ytick.labelsize"]  = 12
plt.rcParams["axes.labelsize"]   = 12
plt.rcParams["axes.titlesize"]   = 12
plt.rcParams["legend.fontsize"]  = 12

linewidth = 0.6

x_pos = np.arange(num_person)
bar_width = 0.7  # unify bar thickness for left & right

# ============================================================
# 6. Create figure with 2x2 grid:
#    left column: top (1st m.) / bottom (2nd m.)
#    right column: single axis spanning both rows (summary)
# ============================================================

fig = plt.figure(figsize=(18.4, 8.2), dpi=100)
gs  = gridspec.GridSpec(
    2,
    2,
    width_ratios=[num_person, 1],
    height_ratios=[h1, h2],
    figure=fig,
)

ax_top    = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_right  = fig.add_subplot(gs[:, 1])  # spans both rows

# ============================================================
# 7. TOP PANEL: 1st measurement, per person
#    Coloring rule for train/test:
#      - train person: usable filled with train_color
#      - test person : usable transparent with border only
#      - complete    : hatched; test person hatch border uses test_color
# ============================================================

bottom = np.zeros(num_person)

for j in range(num_cond):
    usable_cond   = usable_first[:, j]
    complete_cond = complete_first[:, j]

    ax_top.bar(
        x_pos,
        usable_cond,
        bottom=bottom,
        color="none",
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )
    ax_top.bar(
        x_pos,
        usable_cond * is_train_first_person,
        bottom=bottom,
        color=train_color,
        edgecolor="none",
        width=bar_width,
        linewidth=0,
        alpha=0.85,
    )

    complete_facecolors = np.where(
        is_train_first_person, train_color, test_color
    )

    ax_top.bar(
        x_pos,
        complete_cond,
        bottom=bottom,
        color=complete_facecolors,
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
        alpha=0.85,
    )
    ax_top.bar(
        x_pos,
        complete_cond,
        bottom=bottom,
        facecolor="none",
        hatch="////",
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )

    bottom += usable_cond

ax_top.set_ylim(global_ylim)
ax_top.grid(axis="y", alpha=0.6)

# spines: keep left, remove top/right/bottom
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(True)
ax_top.spines["left"].set_visible(True)

ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
ax_top.set_xlim(-0.5, num_person - 0.5)

# label total samples above each bar (1st m.)
for i, total_count in enumerate(first_heights):
    if total_count > 0:
        ax_top.text(
            x_pos[i],
            total_count + ymax * 0.01,
            f"{total_count}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

# ============================================================
# 8. BOTTOM PANEL: 2nd measurement (repeat), per person
# ============================================================

bottom = np.zeros(num_person)

for j in range(num_cond):
    usable_cond   = usable_repeat[:, j]
    complete_cond = complete_repeat[:, j]

    ax_bottom.bar(
        x_pos,
        usable_cond,
        bottom=bottom,
        color="none",
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )
    ax_bottom.bar(
        x_pos,
        usable_cond * is_train_repeat_person,
        bottom=bottom,
        color=train_color,
        edgecolor="none",
        width=bar_width,
        linewidth=0,
        alpha=0.85,
    )
    complete_facecolors = np.where(
        is_train_repeat_person, train_color, test_color
    )
    ax_bottom.bar(
        x_pos,
        complete_cond,
        bottom=bottom,
        color=complete_facecolors,
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
        alpha=0.85,
    )
    ax_bottom.bar(
        x_pos,
        complete_cond,
        bottom=bottom,
        facecolor="none",
        hatch="////",
        edgecolor="black",
        width=bar_width,
        linewidth=linewidth,
    )

    bottom += usable_cond

ax_bottom.set_ylim(global_ylim)
ax_bottom.grid(axis="y", alpha=0.6)

ax_bottom.spines["top"].set_visible(False)
ax_bottom.spines["right"].set_visible(False)
ax_bottom.spines["left"].set_visible(True)
ax_bottom.spines["bottom"].set_visible(True)

ax_bottom.set_xticks(x_pos)
ax_bottom.set_xticklabels(persons, rotation=60, ha="right")

# label total samples above each bar (2nd m.)
for i, total_count in enumerate(repeat_heights):
    if total_count > 0:
        ax_bottom.text(
            x_pos[i],
            total_count + ymax * 0.01,
            f"{total_count}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

# ============================================================
# 9. RIGHT PANEL: one bar for ALL samples (N_total),
# ============================================================

x_bar = np.array([0])

# aggregate totals across all persons and both measurements
total_usable_all   = usable_first.sum() + usable_repeat.sum()
total_complete_all = complete_first.sum() + complete_repeat.sum()

# split usable into non-shadow (usable but not complete) vs 
# shadow (complete)
usable_nonshadow_mask = usable & (~complete)
usable_shadow_mask    = complete

nonshadow_train = int((usable_nonshadow_mask & is_train_sample).sum())
nonshadow_test  = int((usable_nonshadow_mask & is_test_sample).sum())
nonshadow_other = int((usable_nonshadow_mask & is_other_sample).sum())

shadow_train = int((usable_shadow_mask & is_train_sample).sum())
shadow_test  = int((usable_shadow_mask & is_test_sample).sum())
shadow_other = int((usable_shadow_mask & is_other_sample).sum())

nonshadow_total = nonshadow_train + nonshadow_test + nonshadow_other
shadow_total    = shadow_train + shadow_test + shadow_other
assert nonshadow_total + shadow_total == total_usable_all

bottom = 0

# --- shadow (complete): train / test / unused, hatched ---
ax_right.bar(
    x_bar[0],
    shadow_train,
    bottom=bottom,
    color=train_color,
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
    alpha=0.85,
    hatch="////",
)
bottom += shadow_train

ax_right.bar(
    x_bar[0],
    shadow_test,
    bottom=bottom,
    color=test_color,
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
    alpha=0.85,
    hatch="////",
)
bottom += shadow_test

ax_right.bar(
    x_bar[0],
    shadow_other,
    bottom=bottom,
    color="none",
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
    hatch="////",
)
bottom += shadow_other

# --- non-shadow: train colored; test/unused uncolored outlines ---
ax_right.bar(
    x_bar[0],
    nonshadow_train,
    bottom=bottom,
    color=train_color,
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
)
bottom += nonshadow_train

ax_right.bar(
    x_bar[0],
    nonshadow_test,
    bottom=bottom,
    color="none",
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
)
bottom += nonshadow_test

ax_right.bar(
    x_bar[0],
    nonshadow_other,
    bottom=bottom,
    color="none",
    edgecolor="black",
    width=bar_width,
    linewidth=linewidth,
)
bottom += nonshadow_other

usable_bar_total = shadow_total + nonshadow_total
# unify y-tick labels: include shadow/non-shadow totals and train/test
# boundaries
ytick_vals = [
    shadow_train, shadow_total, 
    shadow_total + nonshadow_train, usable_bar_total
]
ytick_vals = sorted(dict.fromkeys(ytick_vals))
ax_right.set_yticks(ytick_vals)

# --- Spine styles ---
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.spines["bottom"].set_visible(True)

# --- X ticks: a single label ---
ax_right.set_xticks(x_bar)
ax_right.set_xticklabels(["All"], fontfamily="Helvetica")

ax_right.tick_params(
    axis="y", which="both", labelright=True, labelleft=False, rotation=-60
)

plt.tight_layout()
print("saved: data/presentation/DataSplit.png")
os.makedirs("data/presentation", exist_ok=True)
plt.savefig(
    "data/presentation/DataSplit.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
plt.close(fig)


# %%
