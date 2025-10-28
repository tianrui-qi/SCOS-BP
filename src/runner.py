import torch
import lightning


__all__ = ["Runner"]


class Runner(lightning.LightningModule):
    weight: torch.Tensor

    def __init__(
        self, model: torch.nn.Module, 
        enable: list[bool], weight: list[float], T: float, 
        lr: float, step_size: int, gamma: float, 
        **kwargs
    ) -> None:
        super().__init__()
        # model
        self.model = model
        # loss
        self.register_buffer("weight", torch.tensor(
            [w for w, e in zip(weight, enable) if e], dtype=torch.float
        ))
        self.task = [f for f, e in zip([
            self._stepContrastive,
            self._stepReconstruction,
            self._stepRegression,
        ], enable) if e]
        self.T = T
        # optimizer
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage='valid')

    def _step(self, batch, stage):
        loss = [f(batch, stage) for f in self.task]
        loss = torch.stack([
            l.detach() if w == 0 else l 
            for l, w in zip(loss, self.weight)
        ])
        loss = torch.dot(loss, self.weight)
        # log
        self.log(
            f"loss/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def _stepContrastive(self, batch, stage):
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
        self.log(
            f"loss/contrastive/{stage}", loss,
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def _stepReconstruction(self, batch, stage):
        x, channel_idx, y = batch   # (B, C, T), (B, C), (B, out_dim)
        x, y = self.model(          # (#mask, S), (#mask, S)
            x, channel_idx,
            masking_type="reconstruction", head_type="reconstruction",
        )
        loss = torch.nn.functional.mse_loss(x, y)
        self.log(
            f"loss/reconstruction/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def _stepRegression(self, batch, stage):
        x, channel_idx, y = batch   # (B, C, T), (B, C), (B, out_dim)
        x, _ = self.model(          # (B, out_dim), None
            x, channel_idx,
            masking_type="contrastive", head_type="regression",
        )
        loss = torch.nn.functional.mse_loss(x, y)
        self.log(
            f"loss/regression/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return [optimizer], [scheduler]
