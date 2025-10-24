import torch


__all__ = ["Transformer"]


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

    def forward(
        self, 
        x: torch.Tensor,                                    # (B, C*L, D), float
        src_key_padding_mask: torch.Tensor | None = None,   # (B, C*L), bool
        pool: bool = True,
    ) -> torch.Tensor:
        x = self.encoder(       # (B, C*L, D) -> (B, C*L, D)
            x, src_key_padding_mask=src_key_padding_mask
        )
        if not pool:
            return x    # (B, C*L, D)
        if src_key_padding_mask is None:
            x = x.mean(dim=1)   # (B, C*L, D) -> (B, D)
            return x    # (B, D)
        if src_key_padding_mask is not None:
            keep = (~src_key_padding_mask).float()                  # (B, C*L)
            denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)    # (B, 1)
            x = (x * keep.unsqueeze(-1)).sum(dim=1) / denom
            return x    # (B, D)
