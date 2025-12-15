import torch


class Transformer(torch.nn.Module):
    def __init__(
        self, D: int, 
        num_layers: int = 4, nhead: int = 8, dim_feedforward: int = 1024,
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
        x: torch.Tensor,            # (B, C*L, D), float
        src_key_padding_mask: (     # (B, C*L), bool
            torch.Tensor | None
        ) = None,
    ) -> torch.Tensor:
        x = self.encoder(   # (B, C*L, D) -> (B, C*L, D)
            x, src_key_padding_mask=src_key_padding_mask
        )
        return x            # (B, C*L, D)
