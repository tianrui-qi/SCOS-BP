import torch
import lightning


__all__ = ["DataModule", "Dataset"]


class DataModule(lightning.LightningDataModule):
    def __init__(
        self, x_load_path: str, y_load_path: str, split_load_path: str,
        y_as_channel: bool,
        channel_perm: bool, channel_drop: float, channel_shift: float,
        batch_size: int, num_workers: int,
    ) -> None:
        super().__init__()
        # path
        self.x_load_path = x_load_path
        self.y_load_path = y_load_path
        self.split_load_path = split_load_path
        # if y_as_channel, append y as an additional channel to x
        self.y_as_channel = y_as_channel
        # dataset
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
        x = torch.load(self.x_load_path, weights_only=True)
        y = torch.load(self.y_load_path, weights_only=True)
        split = torch.load(self.split_load_path, weights_only=True)
        # if y is waveform, normalize to zero mean and unit std
        if x.shape[-1] == y.shape[-1]:
            valid = ~torch.isnan(y).any(dim=1)
            y = (y - y[valid].mean()) / y[valid].std()
        # if y_as_channel, append y as an additional channel to x
        if self.y_as_channel: x = torch.cat([x, y.unsqueeze(1)], dim=1)
        # split
        tr = torch.as_tensor(split == 0, dtype=torch.bool)
        va = torch.as_tensor(split == 1, dtype=torch.bool)
        te = torch.as_tensor(split == 2, dtype=torch.bool)
        # dataset
        self.train_dataset = Dataset(
            x[tr], y[tr], 
            channel_perm=self.channel_perm, 
            channel_drop=self.channel_drop,
            channel_shift=self.channel_shift,
        )
        self.val_dataset   = Dataset(x[va], y[va])
        self.test_dataset  = Dataset(x[te], y[te])

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


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, 
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
    ) -> None:
        # data
        self.x = x  # (N, C, T)
        self.y = y  # (N, T) or (N, out_dim)
        # parameters
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.channel_shift = channel_shift

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[i].clone()
        y = self.y[i].clone()
        # set channel_idx to -1 for channels that are all nan
        channel_idx = torch.arange(len(x), device=x.device, dtype=torch.long)
        channel_idx[torch.all(torch.isnan(x), dim=-1)] = -1
        # if channel_perm, randomly shuffle channels order
        if self.channel_perm:
            perm = torch.randperm(len(channel_idx), device=x.device)
            x, channel_idx = x[perm], channel_idx[perm]
        # if channel_drop, randomly drop some channels by setting channel
        # index to -1 and setting corresponding x to nan; 
        # keep at least one channel
        if torch.rand(()) < self.channel_drop:
            valid_mask = channel_idx != -1
            valid_idx = torch.where(valid_mask)[0]
            # randomly keep k number of channels
            k = torch.randint(1, len(valid_idx)+1, (1,), device=x.device).item()
            perm = torch.randperm(len(valid_idx), device=x.device)
            keep_idx = valid_idx[perm][:k]
            # drop channels that valid but not keep
            drop_mask = torch.zeros_like(channel_idx, dtype=torch.bool)
            drop_mask[valid_idx] = True
            drop_mask[keep_idx] = False
            # update x and channel_idx
            x[drop_mask] = float("nan")
            channel_idx[drop_mask] = -1
        # if channel_shift in (0, 1), shift all channels by random amount
        # in [-channel_shift*T, channel_shift*T)
        # if channel_shift >= 1, shift by random amount in 
        # [-channel_shift, channel_shift]
        # pad nan for the shifted positions
        if self.channel_shift > 0:
            if self.channel_shift < 1:
                max_shift = int(x.shape[-1] * self.channel_shift)
            else:
                max_shift = int(self.channel_shift)
            s = torch.randint(
                -max_shift, max_shift + 1, (1,), device=x.device
            ).item()
            nan = torch.full_like(x, float('nan'))
            if s > 0: nan[..., s:] = x[..., :-s]
            if s < 0: nan[..., :s] = x[..., -s:]
            if s != 0: x = nan
        # after chat with Ariane 23 Oct 2025, we remove this logic since nan 
        # value exist for quality control, not on waveform; there is a qc score
        # for each pulse and it will be nan if the pulse does not pass qc. 
        # if self.p_nan > 0, select a probability between [0, self.p_nan) and 
        # drop each value to nan with that probability
        # if self.p_nan > 0:
        #     p_nan = torch.rand((), device=x.device) * self.p_nan
        #     x = x.masked_fill(torch.rand_like(x) < p_nan, float('nan'))
        return x, channel_idx, y
