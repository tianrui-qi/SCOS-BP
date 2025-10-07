import torch

import src.model.tokenizer
import src.model.masking
import src.model.embedding
import src.model.transformer
import src.model.head


__all__ = ["SCOST"]


class SCOST(torch.nn.Module):
    def __init__(
        self, D: int, S: int, stride: int, C_max: int, L_max: int,
        num_layers: int, nhead: int, dim_feedforward: int, out_dim: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = src.model.tokenizer.Tokenizer(S, stride)
        self.masking = src.model.masking.Masking()
        self.embedding = src.model.embedding.Embedding(D, S, C_max, L_max)
        self.transformer = src.model.transformer.Transformer(
            D, num_layers, nhead, dim_feedforward
        )
        self.head_reconstruction = src.model.head.HeadReconstruction(D, S)
        self.head_regression = src.model.head.HeadRegression(D, out_dim)

    def forward(
        self, x: torch.Tensor,      # (B, C, T)
        channel_idx: torch.Tensor,  # (B, C)
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = self.tokenizer(x)   # (B, C, T) -> (B, C, L, S)
        if self.training:
            mlm_mask, mlm_mode, mlm_rand = self.masking(token)
        else:
            mlm_mask, mlm_mode, mlm_rand = None, None, None
        # (B, C, L, S) -> (B, C, L, D)
        x = self.embedding(token, channel_idx, mlm_mask, mlm_mode, mlm_rand)                           
        B, C, L, D = x.shape
        x = x.reshape((B, C*L, D))  # (B, C, L, D) -> (B, C*L, D)
        x = self.transformer(x)     # (B, C*L, D) -> (B, C*L, D)
        if self.training:
            x = self.head_reconstruction(x)
            x = x.reshape((B, C, L, token.shape[-1]))
            # (B, C, L, S), (B, C, L, S), (B, C, L)
            return x, token, mlm_mask   # type: ignore
        else:
            x = self.head_regression(x)
            # (B, out_dim)
            return x