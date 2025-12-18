import lightning
import torch

from ..model import Model


class ObjectiveFinetune(lightning.LightningModule):
    def __init__(
        self, 
        # model
        model: Model, 
        freeze_embedding: bool, freeze_transformer: int, freeze_head: bool,
        # stepRegressionAdapter, stepAdapter
        K: int = 50,
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
        # stepRegressionAdapter
        self.K = K
        self.weight_shape = weight_shape
        self.weight_min = weight_min
        self.weight_max = weight_max
        # optimizer
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def stepRegressionAdapter(self, batch):
        x, x_channel_idx, y = batch[:3]
        x = (       # (B, T)
            self.model.forwardRegressionAdapter(x, x_channel_idx)
        )
        loss_shape = torch.nn.functional.smooth_l1_loss(    # (B,)
            input=torch.stack([             # (B, 2K+1, [K:-K])
                torch.roll(x, shifts=s, dims=-1) 
                for s in range(-self.K, self.K+1)
            ], dim=1)[:, :, self.K:-self.K],
            target=y.unsqueeze(1).expand(   # (B, 2K+1, [K:-K])
                (-1, 2*self.K+1, -1)
            )[:, :, self.K:-self.K],
            reduction='none'
        ).mean(dim=-1).min(dim=-1).values.mean()
        loss_min   = torch.nn.functional.smooth_l1_loss(    # (B,)
            x.min(dim=-1).values, y.min(dim=-1).values
        )
        loss_max   = torch.nn.functional.smooth_l1_loss(    # (B,)
            x.max(dim=-1).values, y.max(dim=-1).values
        )
        loss = (
            self.weight_shape * loss_shape +
            self.weight_min * loss_min +
            self.weight_max * loss_max
        )
        return loss, loss_shape, loss_min, loss_max

    def stepAdapter(self, batch):
        x, y = batch[:2]
        x = (       # (B, T)
            self.model.forwardAdapter(x)
        )
        loss_shape = torch.nn.functional.smooth_l1_loss(    # (B,)
            input=torch.stack([             # (B, 2K+1, [K:-K])
                torch.roll(x, shifts=s, dims=-1) 
                for s in range(-self.K, self.K+1)
            ], dim=1)[:, :, self.K:-self.K],
            target=y.unsqueeze(1).expand(   # (B, 2K+1, [K:-K])
                (-1, 2*self.K+1, -1)
            )[:, :, self.K:-self.K],
            reduction='none'
        ).mean(dim=-1).min(dim=-1).values.mean()
        loss_min   = torch.nn.functional.smooth_l1_loss(    # (B,)
            x.min(dim=-1).values, y.min(dim=-1).values
        )
        loss_max   = torch.nn.functional.smooth_l1_loss(    # (B,)
            x.max(dim=-1).values, y.max(dim=-1).values
        )
        loss = (
            self.weight_shape * loss_shape +
            self.weight_min * loss_min +
            self.weight_max * loss_max
        )
        return loss, loss_shape, loss_min, loss_max

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return [optimizer], [scheduler]
