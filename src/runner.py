import torch
import lightning


__all__ = ["Pretrain"]


class Pretrain(lightning.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y, mask = self.model(    # (B, C, L, S), (B, C, L, S), (B, C, L)
            x, channel_idx, mask=True, task="reconstruction"
        )
        # calculate loss only on masked tokens
        loss = torch.nn.functional.mse_loss(x[mask], y[mask])
        self.log(
            "Pretrain/train/loss", loss, 
            on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y, mask = self.model(    # (B, C, L, S), (B, C, L, S), (B, C, L)
            x, channel_idx, mask=True, task="reconstruction"
        )
        loss = torch.nn.functional.mse_loss(x[mask], y[mask])
        self.log(
            "Pretrain/valid/loss-mask", loss, 
            on_epoch=True, logger=True, prog_bar=True
        )
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y, _ = self.model(       # (B, C, T), (B, C, T), None
            x, channel_idx, mask=False, task="reconstruction"
        )
        loss = torch.nn.functional.mse_loss(x, y)
        self.log(
            "Pretrain/valid/loss-all", loss, 
            on_epoch=True, logger=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y, mask = self.model(    # (B, C, L, S), (B, C, L, S), (B, C, L)
            x, channel_idx, mask=True, task="reconstruction"
        )
        loss = torch.nn.functional.mse_loss(x[mask], y[mask])
        self.log(
            "Pretrain/valid/loss-mask", loss, 
            on_epoch=True, logger=True, prog_bar=True
        )
        x, channel_idx, _ = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y, _ = self.model(       # (B, C, T), (B, C, T), None
            x, channel_idx, mask=False, task="reconstruction"
        )
        loss = torch.nn.functional.mse_loss(x, y)
        self.log(
            "Pretrain/valid/loss-all", loss, 
            on_epoch=True, logger=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        return [optimizer], [scheduler]
