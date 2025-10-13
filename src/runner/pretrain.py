import torch
import lightning


__all__ = ["Pretrain"]


class Pretrain(lightning.LightningModule):
    def __init__(
        self, model: torch.nn.Module, 
        lr: float, step_size: int, gamma: float, **kwargs
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = torch.nn.functional.mse_loss
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y = self.model(          # (B, C, L, S), (B, C, L, S)
            x, channel_idx, mask=True, task="reconstruction"
        )
        loss = self.loss(x, y)
        self.log(
            "loss/train", loss, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y = self.model(          # (B, C, L, S), (B, C, L, S)
            x, channel_idx, mask=True, task="reconstruction"
        )
        loss = self.loss(x, y)
        self.log(
            "loss/valid", loss, on_step=False, on_epoch=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y = self.model(          # (B, C, L, S), (B, C, L, S)
            x, channel_idx, mask=True, task="reconstruction"
        )
        loss = self.loss(x, y)
        self.log(
            "loss/test", loss, on_step=False, on_epoch=True, logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return [optimizer], [scheduler]
