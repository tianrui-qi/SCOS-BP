import torch
import lightning

from ..model import SCOST


class Pretrain(lightning.LightningModule):
    def __init__(
        self, 
        # model
        model: SCOST, 
        freeze_embedding: bool, freeze_transformer: int, freeze_head: bool,
        # loss
        enable: tuple[bool, bool, bool], weight: tuple[float, float, float], 
        # loss: contrastive
        T: float = 0.2,
        # loss: reconstruction
        p_point: float = 0.2, 
        p_span_small: tuple[float, float] = (0.0, 0.5),
        p_span_large: tuple[float, float] = (0.0, 1.0),
        p_hide: float = 0.9, p_keep: float = 0.1,
        # optimizer
        lr: float = 0.005, step_size: int = 30, gamma: float = 0.98, 
        **kwargs,
    ) -> None:
        super().__init__()
        # model
        self.model = model
        self.model.freeze(freeze_embedding, freeze_transformer, freeze_head)
        # loss
        self.register_buffer("weight", torch.tensor(
            [w for w, e in zip(weight, enable) if e], dtype=torch.float
        ))
        self.task = [f for f, e in zip([
            self._stepContrastive,
            self._stepReconstruction,
            self._stepRegression,
        ], enable) if e]
        # loss: contrastive 
        self.T = T
        # loss: reconstruction
        self.p_point = p_point
        self.p_span_small = p_span_small
        self.p_span_large = p_span_large
        self.p_hide = p_hide
        self.p_keep = p_keep
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
        x, x_channel_idx, _ = batch
        x_pred = (      # (B, D)
            self.model.forwardContrastive(x, x_channel_idx)
        )
        x_orig = (      # (B, D)
            self.model.forward(x, x_channel_idx)
        ).detach()
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
        x, x_channel_idx, _ = batch
        x, y = (    # (#mask, S), (#mask, S)
            self.model.forwardReconstruction(
                x, x_channel_idx,
                p_point=self.p_point,
                p_span_small=self.p_span_small,
                p_span_large=self.p_span_large,
                p_hide=self.p_hide,
                p_keep=self.p_keep,
            )
        )
        loss = torch.nn.functional.smooth_l1_loss(x, y)
        self.log(
            f"loss/reconstruction/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def _stepRegression(self, batch, stage):
        x, channel_idx, y = batch
        x = (           # (B, out_dim)
            self.model.forwardRegression(x, channel_idx)
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
