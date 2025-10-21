import torch
import lightning
import pandas as pd


__all__ = ["DataModule"]


class DataModule(lightning.LightningDataModule):
    def __init__(
        self, x_load_path: str, y_load_path: str, profile_load_path: str,
        channel_perm: bool, channel_drop: bool, p_nan: float, 
        batch_size: int, num_workers: int,
    ) -> None:
        super().__init__()
        # load
        self.x_load_path = x_load_path
        self.y_load_path = y_load_path
        self.profile_load_path = profile_load_path
        # dataset
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.p_nan = p_nan
        # dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

    def setup(self, stage=None) -> None:
        # load
        x = torch.load(self.x_load_path, weights_only=True)
        y = torch.load(self.y_load_path, weights_only=True)
        profile = pd.read_csv(self.profile_load_path)
        # split
        split = profile["pretrain"].to_numpy()
        tr = torch.as_tensor(split == 0, dtype=torch.bool)
        va = torch.as_tensor(split == 1, dtype=torch.bool)
        te = torch.as_tensor(split == 2, dtype=torch.bool)
        # dataset
        self.train_dataset = Dataset(
            x[tr], y[tr], 
            channel_perm=self.channel_perm, channel_drop=self.channel_drop,
            p_nan=self.p_nan
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
        channel_perm: bool = False, channel_drop: bool = False, 
        p_nan: float = 0.0,
    ) -> None:
        self.x, self.y = x, y   # (N, C, T), (N, out_dim)
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.p_nan = p_nan

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[i].clone()
        channel_idx = torch.arange(len(x), device=x.device, dtype=torch.long)
        y = self.y[i].clone()
        # if channel_perm, randomly shuffle channels order
        if self.channel_perm:
            perm = torch.randperm(len(channel_idx), device=x.device)
            x, channel_idx = x[perm], channel_idx[perm]
        # if channel_drop, randomly drop some channels by setting channel
        # index to -1 and setting corresponding x to nan; keep at least one 
        # channel
        if self.channel_drop:
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
        # if self.p_nan > 0, select a probability between [0, self.p_nan) and 
        # drop each value to nan with that probability
        if self.p_nan > 0:
            p_nan = torch.rand((), device=x.device) * self.p_nan
            x = x.masked_fill(torch.rand_like(x) < p_nan, float('nan'))
        return x, channel_idx, y
