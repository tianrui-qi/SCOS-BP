import torch
import lightning


__all__ = ["Runner"]


class Runner(lightning.LightningModule):
    def __init__(
        self, model: torch.nn.Module, weight: list[float],
        T: float, lr: float, step_size: int, gamma: float, **kwargs
    ) -> None:
        super().__init__()
        self.model = model
        self.weight = weight
        self.T = T
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        loss1 = self._step_reconstruction(batch)
        loss2 = self._step_contrastive(batch)
        loss3 = self._step_regression(batch)
        loss4 = self._step_regression_mask(batch)
        # total loss
        loss = torch.stack([loss1, loss2, loss3, loss4])
        weights = torch.as_tensor(self.weight, device=loss1.device)
        loss = (loss * weights).sum()
        # log
        self.log(
            "loss/train", loss, on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_reconstruction/train", loss1, 
            on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_contrastive/train", loss2, 
            on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_regression/train", loss3, 
            on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_regression_mask/train", loss4, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss1 = self._step_reconstruction(batch)
        loss2 = self._step_contrastive(batch)
        loss3 = self._step_regression(batch)
        loss4 = self._step_regression_mask(batch)
        # total loss
        loss = torch.stack([loss1, loss2, loss3, loss4])
        weights = torch.as_tensor(self.weight, device=loss1.device)
        loss = (loss * weights).sum()
        # log
        self.log(
            "loss/valid", loss, on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_reconstruction/valid", loss1, 
            on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_contrastive/valid", loss2, 
            on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_regression/valid", loss3, 
            on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "loss_regression_mask/valid", loss4, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def _step_reconstruction(self, batch):
        x, channel_idx, y = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y = self.model(          # (#mask, S), (#mask, S)
            x, channel_idx,
            masking_type="reconstruction", head_type="reconstruction",
        )
        loss = torch.nn.functional.mse_loss(x, y)
        return loss
    
    def _step_contrastive(self, batch):
        x, channel_idx, y = batch   # (B, C, T), (B, C), (B, out_dim)
        x_pred, _ = self.model(     # (B, D), None
            x, channel_idx,
            masking_type="contrastive", head_type="contrastive",
        )
        x_orig, _ = self.model(     # (B, D), None
            x, channel_idx,
            masking_type=None, pool=True,
        )
        x_orig = x_orig.detach()
        x_pred = torch.nn.functional.normalize(x_pred, dim=1, p=2)
        x_orig = torch.nn.functional.normalize(x_orig, dim=1, p=2)
        labels = torch.arange(x_orig.shape[0], device=x_orig.device)
        logits_op = (x_orig @ x_pred.t()) / self.T
        logits_po = (x_pred @ x_orig.t()) / self.T
        loss = (
            torch.nn.functional.cross_entropy(logits_op, labels) +
            torch.nn.functional.cross_entropy(logits_po, labels)
        ) * 0.5
        return loss

    def _step_regression(self, batch):
        x, channel_idx, y = batch   # (B, C, T), (B, C), (B, out_dim)
        x, _ = self.model(          # (B, out_dim), None
            x, channel_idx,
            masking_type=None, head_type="regression",
        )
        loss = torch.nn.functional.mse_loss(x, y)
        return loss

    def _step_regression_mask(self, batch):
        x, channel_idx, y = batch   # (B, C, T), (B, C), (B, out_dim)
        x, _ = self.model(          # (B, out_dim), None
            x, channel_idx,
            masking_type="contrastive", head_type="regression",
        )
        loss = torch.nn.functional.mse_loss(x, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return [optimizer], [scheduler]
