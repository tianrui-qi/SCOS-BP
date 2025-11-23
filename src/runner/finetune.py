import torch
import lightning

from ..model import SCOST


class Finetune(lightning.LightningModule):
    def __init__(
        self, 
        # model
        model: SCOST, 
        freeze_embedding: bool, freeze_transformer: int, freeze_head: bool,
        # loss
        weight_shape: float = 0.2, 
        weight_min: float = 0.4, weight_max: float = 0.4,
        # optimizer
        lr: float = 0.005, step_size: int = 30, gamma: float = 0.98, 
        **kwargs,
    ) -> None:
        super().__init__()
        # model
        self.model = model
        self.model.freeze(freeze_embedding, freeze_transformer, freeze_head)
        # loss
        self.weight_shape = weight_shape
        self.weight_min = weight_min
        self.weight_max = weight_max
        # optimizer
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage='valid')

    def _step(self, batch, stage):
        x, channel_idx, y = batch
        x = (           # (B, T)
            self.model.forwardRegression(x, channel_idx, adapter=True)
        )
        loss_shape = torch.nn.functional.mse_loss(x, y)     # (B, T)
        loss_min   = torch.nn.functional.mse_loss(          # (B,)
            x.min(dim=-1).values, y.min(dim=-1).values
        )
        loss_max   = torch.nn.functional.mse_loss(          # (B,)
            x.max(dim=-1).values, y.max(dim=-1).values
        )
        loss = (
            self.weight_shape * loss_shape +
            self.weight_min * loss_min +
            self.weight_max * loss_max
        )
        # log
        self.log(
            f"loss/shape/{stage}", loss_shape,
            on_step=False, on_epoch=True, logger=True,
        )
        self.log(
            f"loss/min/{stage}", loss_min,
            on_step=False, on_epoch=True, logger=True,
        )
        self.log(
            f"loss/max/{stage}", loss_max,
            on_step=False, on_epoch=True, logger=True,
        )
        self.log(
            f"loss/{stage}", loss,
            on_step=False, on_epoch=True, logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return [optimizer], [scheduler]
