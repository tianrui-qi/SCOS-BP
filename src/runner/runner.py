import torch
import lightning

from ..model import SCOST


class Runner(lightning.LightningModule):
    def __init__(
        self, 
        model: SCOST, 
        freeze_embedding: bool, freeze_transformer: int,
        enable: tuple[bool, ...], weight: tuple[float, ...], 
        T: float = 0.2,
        p_point: float = 0.2, 
        p_span_small: tuple[float, float] = (0.0, 0.5),
        p_span_large: tuple[float, float] = (0.0, 1.0),
        p_hide: float = 0.9, p_keep: float = 0.1,
        lr: float = 0.005, step_size: int = 20, gamma: float = 0.98, 
    ) -> None:
        super().__init__()
        # model
        self.model = model
        self.model.freeze(freeze_embedding, freeze_transformer)
        # loss
        self.register_buffer("weight", torch.tensor(
            [w for w, e in zip(weight, enable) if e], dtype=torch.float
        ))
        self.task = [f for f, e in zip([
            self._stepContrastive,
            self._stepReconstructionCal,
            self._stepReconstructionRaw,
            self._stepRegression,
        ], enable) if e]
        # contrastive 
        self.T = T
        # reconstruction
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
        loss = [f(b, stage) for f, b in zip(self.task, batch)]
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
        x, x_channel_idx = batch
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

    def _stepReconstructionCal(self, batch, stage):
        x, x_channel_idx, c, c_channel_idx, y = batch
        if stage == "train":
            x, y = (    # (#mask, S), (#mask, S)
                self.model.forwardReconstructionCal(
                    x, x_channel_idx, c, c_channel_idx, y,
                    p_point=self.p_point,
                    p_span_small=self.p_span_small,
                    p_span_large=self.p_span_large,
                    p_hide=self.p_hide,
                    p_keep=self.p_keep,
                )
            )
            loss = torch.nn.functional.smooth_l1_loss(x, y)
        else:
            x, y = (    # (B, C, T), (B, C, T)
                self.model.forwardReconstructionCal(
                    x, x_channel_idx, c, c_channel_idx, y,
                    user_mask=3,
                )
            )
            true_min = y[:, 3, :].min(dim=-1).values
            true_max = y[:, 3, :].max(dim=-1).values
            pred_min = x[:, 3, :].min(dim=-1).values
            pred_max = x[:, 3, :].max(dim=-1).values
            mask = ~(
                torch.isnan(true_min) | torch.isnan(true_max) | 
                torch.isnan(pred_min) | torch.isnan(pred_max)
            )
            true_min = true_min[mask]
            true_max = true_max[mask]
            pred_min = pred_min[mask]
            pred_max = pred_max[mask]
            loss = (
                torch.nn.functional.l1_loss(pred_min, true_min) +
                torch.nn.functional.l1_loss(pred_max, true_max)
            ) * 0.5 if mask.sum() > 0 else x.new_tensor(0.0)
        self.log(
            f"loss/reconstruction/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def _stepReconstructionRaw(self, batch, stage):
        x, x_channel_idx = batch
        if stage == "train":
            x, y = (    # (#mask, S), (#mask, S)
                self.model.forwardReconstructionRaw(
                    x, x_channel_idx,
                    p_point=self.p_point,
                    p_span_small=self.p_span_small,
                    p_span_large=self.p_span_large,
                    p_hide=self.p_hide,
                    p_keep=self.p_keep,
                )
            )
            loss = torch.nn.functional.smooth_l1_loss(x, y)
        else:
            x, y = (    # (B, C, T), (B, C, T)
                self.model.forwardReconstructionRaw(
                    x, x_channel_idx,
                    user_mask=3,
                )
            )
            true_min = y[:, 3, :].min(dim=-1).values
            true_max = y[:, 3, :].max(dim=-1).values
            pred_min = x[:, 3, :].min(dim=-1).values
            pred_max = x[:, 3, :].max(dim=-1).values
            mask = ~(
                torch.isnan(true_min) | torch.isnan(true_max) | 
                torch.isnan(pred_min) | torch.isnan(pred_max)
            )
            true_min = true_min[mask]
            true_max = true_max[mask]
            pred_min = pred_min[mask]
            pred_max = pred_max[mask]
            loss = (
                torch.nn.functional.l1_loss(pred_min, true_min) +
                torch.nn.functional.l1_loss(pred_max, true_max)
            ) * 0.5 if mask.sum() > 0 else x.new_tensor(0.0)
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
