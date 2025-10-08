import torch
import lightning
import pandas as pd


__all__ = ["DataModule"]


class DataModule(lightning.LightningDataModule):
    def __init__(
        self, x_path: str, y_path: str, profile_path: str,
        p_nan: float, batch_size: int, num_workers: int,
    ) -> None:
        super().__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.profile_path = profile_path
        self.p_nan = p_nan
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

    def setup(self, stage=None):
        # load
        x = torch.load(self.x_path)
        y = torch.load(self.y_path)
        profile = pd.read_csv(self.profile_path)
        # split
        split = profile["pretrain"].to_numpy()
        tr = torch.as_tensor(split == 0, dtype=torch.bool)
        va = torch.as_tensor(split == 1, dtype=torch.bool)
        te = torch.as_tensor(split == 2, dtype=torch.bool)
        # dataset
        self.train_dataset = Dataset(
            x[tr], y[tr], shuffle_channel=True, p_nan=self.p_nan
        )
        self.val_dataset = Dataset(
            x[va], y[va], shuffle_channel=False, p_nan=0
        )
        self.test_dataset = Dataset(
            x[te], y[te], shuffle_channel=False, p_nan=0
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, x: torch.Tensor, y: torch.Tensor,
        shuffle_channel: bool = False, p_nan: float = 0.0,
    ):
        self.x, self.y = x, y   # (N, C, T), (N, out_dim)
        self.shuffle_channel = shuffle_channel
        self.p_nan = p_nan

    def __len__(self):
        return len(self.x)

    def __getitem__(
        self, i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[i]
        channel_idx = torch.arange(len(x))
        y = self.y[i]
        # if shuffle_channel, randomly permute the channel dimension
        if self.shuffle_channel:
            channel_idx = channel_idx[torch.randperm(len(x))]
            x = x[channel_idx]
        # if p_nan > 0, randomly set some value to nan
        if self.p_nan > 0:
            p_nan = torch.rand((), device=x.device) * self.p_nan
            x = x.masked_fill(torch.rand_like(x) < p_nan, float('nan'))
        return x, channel_idx, y
