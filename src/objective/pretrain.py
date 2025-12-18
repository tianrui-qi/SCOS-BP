from typing import Literal

import lightning
import torch

from ..model import Model


class ObjectivePretrain(lightning.LightningModule):
    weight_train: torch.Tensor
    weight_validation: torch.Tensor

    def __init__(
        self, 
        # model
        model: Model, 
        freeze_embedding: bool, freeze_transformer: int, freeze_head: bool,
        # step
        task: dict[Literal[
            "stepContrastive", "stepReconstruction", "stepRegression", 
        ], float | None], 
        # stepContrastive
        T: float = 0.2,
        # stepReconstruction
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
        # step
        self.step_train = [
            getattr(self, t) 
            for t in task.keys() 
            if task[t] is not None and task[t] > 0      # type: ignore
        ]
        self.step_validation = [
            getattr(self, t) 
            for t in task.keys() 
            if task[t] is not None and task[t] >= 0     # type: ignore
        ]
        self.register_buffer("weight_train", torch.tensor([
            task[t] 
            for t in task.keys() 
            if task[t] is not None and task[t] > 0      # type: ignore
        ], dtype=torch.float))
        self.register_buffer("weight_validation", torch.tensor([
            task[t] 
            for t in task.keys() 
            if task[t] is not None and task[t] >= 0     # type: ignore
        ], dtype=torch.float))
        # stepContrastive
        self.T = T
        # stepReconstruction
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
        stage='train'
        loss = [f(batch, stage) for f in self.step_train]
        loss = torch.stack([l for l in loss])
        loss = torch.dot(loss, self.weight_train)
        # log
        self.log(
            f"loss/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        stage='valid'
        loss = [f(batch, stage) for f in self.step_validation]
        loss = torch.stack([l for l in loss])
        loss = torch.dot(loss, self.weight_validation)
        # log
        self.log(
            f"loss/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def stepContrastive(self, batch, stage):
        x, x_channel_idx = batch[:2]
        x_pred = (  # (B, D)
            self.model.forwardContrastive(x, x_channel_idx)
        )
        x_orig = (  # (B, D)
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
            f"loss/stepContrastive/{stage}", loss,
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def stepReconstruction(self, batch, stage):
        x, x_channel_idx = batch[:2]
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
            f"loss/stepReconstruction/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def stepRegression(self, batch, stage):
        x, x_channel_idx, y = batch[:3]
        x = (       # (B, out_dim)
            self.model.forwardRegression(x, x_channel_idx)
        )
        loss = torch.nn.functional.mse_loss(x, y)
        self.log(
            f"loss/stepRegression/{stage}", loss, 
            on_step=False, on_epoch=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return [optimizer], [scheduler]
