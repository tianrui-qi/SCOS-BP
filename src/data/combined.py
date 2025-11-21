import lightning
import lightning.pytorch.utilities.combined_loader

from .raw import DataModuleRaw
from .cal import DataModuleCal
from .reg import DataModuleReg


class DataModuleCombined(lightning.LightningDataModule):
    def __init__(self, enable: tuple[bool, ...], **kwargs) -> None:
        super().__init__()
        self.dm_enable = enable
        self.dm_module = [
            DataModuleRaw,  # Contrastive
            DataModuleCal,  # ReconstructionCal
            DataModuleRaw,  # ReconstructionRaw
            DataModuleReg,  # Regression
        ]
        self.dm = [
            m(**kwargs) if e else None 
            for e, m in zip(self.dm_enable, self.dm_module)
        ]

    def setup(self, stage=None):
        for d in self.dm: 
            if d is not None: d.setup(stage)

    def train_dataloader(self):
        return lightning.pytorch.utilities.combined_loader.CombinedLoader(
            [d.train_dataloader() for d in self.dm if d is not None]
        )

    def val_dataloader(self):
        return lightning.pytorch.utilities.combined_loader.CombinedLoader(
            [d.val_dataloader() for d in self.dm if d is not None]
        )
