import torch
import lightning
import pandas as pd

import os


__all__ = ["RawDataModule", "RawDataset"]


class RawDataModule(lightning.LightningDataModule):
    def __init__(
        self, 
        data_load_fold: str, split: str,
        # augment
        channel_perm: bool, channel_drop: float, channel_shift: float,
        # dataloader
        batch_size: int, num_workers: int,
        **kwargs
    ) -> None:
        super().__init__()
        self.x_load_path = os.path.join(data_load_fold, 'x.pt')
        self.sample_load_path = os.path.join(data_load_fold, 'sample.csv')
        self.split = split
        # augment
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.channel_shift = channel_shift
        # dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

    def setup(self, stage=None) -> None:
        # load
        x = torch.load(self.x_load_path, weights_only=True)     # (N, C, T)
        sample = pd.read_csv(self.sample_load_path)
        # normalize each channel
        mean = torch.nanmean(x, dim=(0, 2))
        mean2 = torch.nanmean(x * x, dim=(0, 2))
        std = torch.sqrt(mean2 - mean * mean)
        x = (x - mean.view(1, -1, 1)) / std.view(1, -1, 1)
        # dataset
        self.train_dataset = RawDataset(
            x[sample[self.split] == 0],
            channel_perm=self.channel_perm, 
            channel_drop=self.channel_drop,
            channel_shift=self.channel_shift,
        )
        self.val_dataset   = RawDataset(x[sample[self.split] == 1])
        self.test_dataset  = RawDataset(x[sample[self.split] == 2])

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )


class RawDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        x: torch.Tensor,        # (N, C, T)
        # augment
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
    ) -> None:
        # data
        self.x = x              # (N, C, T)
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
        # augment x
        x, x_channel_idx = RawDataset.augment(
            x, x_channel_idx, 
            channel_perm=self.channel_perm, 
            channel_drop=self.channel_drop, 
            channel_shift=self.channel_shift
        )
        # return
        return x, x_channel_idx

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
            k = torch.randint(1, len(valid_idx)+1, (1,), device=x.device).item()
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
