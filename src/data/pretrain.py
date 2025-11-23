import torch

from .data import DataModule, DataSet


class Pretrain(DataModule):
    def __init__(
        self,
        # dataset
        channel_perm: bool = False, 
        channel_drop: float = 0, 
        channel_shift: float = 0,
        # dataloader
        batch_size: int = 1, num_workers: int = 0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # dataset
        self.channel_perm = channel_perm
        self.channel_drop = channel_drop
        self.channel_shift = channel_shift
        # dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = DataSet(
            self.x[(self.profile["split"] == 0).to_numpy()],
            self.y[(self.profile["split"] == 0).to_numpy()] 
            if self.y_as_y else None,
            channel_perm=self.channel_perm, 
            channel_drop=self.channel_drop,
            channel_shift=self.channel_shift,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
        return dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = DataSet(
            self.x[(self.profile["split"] == 1).to_numpy()], 
            self.y[(self.profile["split"] == 1).to_numpy()]
            if self.y_as_y else None
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
        return dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = DataSet(
            self.x, 
            self.y 
            if self.y_as_y else None,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, 
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
        return dataloader
