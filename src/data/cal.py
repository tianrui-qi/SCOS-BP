import torch
import pandas as pd

from .data import DataModule

from typing import Literal


class DataModuleCal(DataModule):
    def __init__(
        self, 
        P: int = 32,
        channel_idx_bp: int = 3,
        a_range: tuple[float, float] | float | None = None,
        b_range: tuple[float, float] | float | None = None,
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.P = P
        self.channel_idx_bp = channel_idx_bp
        self.a_range = a_range
        self.b_range = b_range
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.channel_shift = channel_shift

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        # dataset
        self.train_dataset = DatasetCal(
            self.x[(self.profile["split"] == 0).to_numpy()], 
            self.profile[self.profile["split"] == 0],
            channel_perm=self.channel_perm, 
            channel_drop=self.channel_drop,
            channel_shift=self.channel_shift,
            P=self.P, 
            channel_idx_bp=self.channel_idx_bp,
            a_range=self.a_range, 
            b_range=self.b_range,
        )
        self.val_dataset   = DatasetCal(
            self.x[(self.profile["split"] == 1).to_numpy()], 
            self.profile[self.profile["split"] == 1],
            P=self.P,
            channel_idx_bp=self.channel_idx_bp,
        )
        self.test_dataset  = DatasetCal(
            self.x,
            self.profile,
            P=self.P,
            channel_idx_bp=self.channel_idx_bp,
        )

    def filter_(
        self, filter_level: Literal["X", "Y", "XY", "All"] | None = None
    ) -> None:
        super().filter_(filter_level=filter_level)
        # initialize
        # remove all samples that do not have valid bp channel
        valid = ~torch.isnan(self.x[:, self.channel_idx_bp]).all(dim=1)
        self.x = self.x[valid]
        self.profile = self.profile.iloc[valid.numpy()].reset_index(drop=True)
        # calibration data group by subject
        # note that invalid subject will not include in self.c
        c = {
            s: x_s 
            for s, sample_s in (
                self.profile[self.profile["condition"] == 1].groupby("subject")
            )
            for x_s in [self.x[sample_s.index.values]]
            if not torch.isnan(x_s[:, self.channel_idx_bp]).all(dim=1).all()
        }
        # for subject not in c, remove that from x, sample
        valid = self.profile["subject"].isin(c.keys()).to_numpy()
        self.x = self.x[valid]
        self.profile = self.profile.iloc[valid].reset_index(drop=True)


class DatasetCal(torch.utils.data.Dataset):
    def __init__(
        self, 
        x: torch.Tensor,    # (N, C, T)
        profile: pd.DataFrame, 
        # perturb
        P: int = 32,
        channel_idx_bp: int = 3,
        a_range: tuple[float, float] | float | None = None,
        b_range: tuple[float, float] | float | None = None,
        # augment
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
    ) -> None:
        # data
        self.x = x          # (N, C, T)
        self.c = {}
        self.profile = profile
        # perturb
        self.P = P
        self.channel_idx_bp = channel_idx_bp
        self.a_range = a_range
        self.b_range = b_range
        # augment
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.channel_shift = channel_shift

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        # x
        x = self.x[i].clone()   # (C, T)
        x_channel_idx = (       # (C,)
            torch.arange(len(x), device=x.device, dtype=torch.long)
        )
        x_channel_idx[torch.all(torch.isnan(x), dim=-1)] = -1
        # c
        c = self.c[self.profile.loc[i, "subject"]].clone()
        c = c[torch.randint(    # (P, C, T)
            low=0, high=len(c), size=(self.P,), device=x.device
        )]
        c_channel_idx = (       # (P, C)
            torch.arange(c.shape[1], device=x.device, dtype=torch.long)
            .unsqueeze(0).expand(self.P, -1)
        )
        c_channel_idx[torch.all(torch.isnan(c), dim=-1)] = -1
        # perturb
        y = x.clone()
        y_bp, a, b = DatasetCal.perturb(
            x[self.channel_idx_bp].unsqueeze(0),
            a_range=self.a_range,
            b_range=self.b_range,
        )
        y[self.channel_idx_bp] = y_bp.squeeze(0)
        c[:, self.channel_idx_bp, :], _, _ = DatasetCal.perturb(
            c[:, self.channel_idx_bp, :],
            a_range=a,
            b_range=b,
        )
        x[self.channel_idx_bp], _, _ = DatasetCal.perturb(
            x[self.channel_idx_bp].unsqueeze(0),
            a_range=self.a_range,
            b_range=self.b_range,
        )
        # augment
        # don't augment x since we need x to be aligned with y
        # can drop channel for c since in reality our data for calibration
        # may not have all channels
        for p in range(self.P):
            c[p], c_channel_idx[p] = DatasetCal.augment(
                c[p], c_channel_idx[p], 
                channel_perm=self.channel_perm, 
                channel_drop=self.channel_drop, 
            )
        # shift x, c, y together
        if self.channel_shift > 0:
            if self.channel_shift < 1:
                max_shift = int(x.shape[-1] * self.channel_shift)
            else:
                max_shift = int(self.channel_shift)
            s = torch.randint(
                -max_shift, max_shift + 1, (1,), device=x.device
            ).item()
            nan_x = torch.full_like(x, float('nan'))    # (C, T)
            nan_y = torch.full_like(y, float('nan'))    # (C, T)
            nan_c = torch.full_like(c, float('nan'))    # (P, C, T)
            if s > 0:
                nan_x[..., s:] = x[..., :-s]
                nan_y[..., s:] = y[..., :-s]
                nan_c[..., s:] = c[..., :-s]
            if s < 0:
                nan_x[..., :s] = x[..., -s:]
                nan_y[..., :s] = y[..., -s:]
                nan_c[..., :s] = c[..., -s:]
            if s != 0:
                x = nan_x
                y = nan_y
                c = nan_c
        # return
        return x, x_channel_idx, c, c_channel_idx, y

    @staticmethod
    def perturb(
        x: torch.Tensor,    # (B, T)
        a_range: tuple[float, float] | float | None = None,
        b_range: tuple[float, float] | float | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        if a_range is None or b_range is None:
            a = 1.0
            b = 0.0
        elif isinstance(a_range, float) and isinstance(b_range, float): 
            a = a_range
            b = b_range
        elif isinstance(a_range, tuple) and isinstance(b_range, tuple):
            a = float(torch.rand(()) * (a_range[1] - a_range[0]) + a_range[0])
            b = float(torch.rand(()) * (b_range[1] - b_range[0]) + b_range[0])
        else:
            raise ValueError
        y = a * x + b   # (B, T)
        return y, a, b  # (B, T), float, float

    @staticmethod
    def augment(
        x: torch.Tensor,                # (C, T)
        x_channel_idx: torch.Tensor,    # (C,)
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C, T = x.shape
        device = x.device

        # if channel_perm, randomly shuffle channels order
        if channel_perm:
            perm = torch.randperm(C, device=device)
            x, x_channel_idx = x[perm], x_channel_idx[perm]

        # if channel_drop, randomly drop some channels by setting channel
        # index to -1 and setting corresponding x to nan; 
        # keep at least one channel
        if torch.rand(()) < channel_drop:
            valid_mask = x_channel_idx != -1
            valid_idx = torch.where(valid_mask)[0]
            # randomly keep k number of channels
            k = torch.randint(
                1, len(valid_idx)+1, (1,), device=x.device
            ).item()
            perm = torch.randperm(len(valid_idx), device=x.device)
            keep_idx = valid_idx[perm][:k]
            # drop channels that valid but not keep
            drop_mask = torch.zeros_like(x_channel_idx, dtype=torch.bool)
            drop_mask[valid_idx] = True
            drop_mask[keep_idx] = False
            # update x and channel_idx
            x[drop_mask] = float("nan")
            x_channel_idx[drop_mask] = -1

        # if channel_shift in (0, 1), shift all channels by random amount
        # in [-channel_shift*T, channel_shift*T)
        # if channel_shift >= 1, shift by random amount in 
        # [-channel_shift, channel_shift]
        # pad nan for the shifted positions
        if channel_shift > 0:
            if channel_shift < 1:
                max_shift = int(T * channel_shift)
            else:
                max_shift = int(channel_shift)
            s = torch.randint(
                -max_shift, max_shift + 1, (1,), device=x.device
            ).item()
            nan = torch.full_like(x, float('nan'))
            if s > 0: nan[..., s:] = x[..., :-s]
            if s < 0: nan[..., :s] = x[..., -s:]
            if s != 0: x = nan

        return x, x_channel_idx
