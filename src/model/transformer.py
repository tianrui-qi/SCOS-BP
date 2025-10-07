import torch


__all__ = []


class Transformer(torch.nn.Module):
    def __init__(
        self, D: int, num_layers: int, nhead: int, dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=D,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ), num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, C*L, D) -> (B, C*L, D)