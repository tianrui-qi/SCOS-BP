import torch

from typing import Literal

from .tokenizer import Tokenizer
from .masking import Masking
from .embedding import Embedding
from .transformer import Transformer
from .head import HeadReconstruction, HeadRegression


__all__ = ["SCOST"]


class SCOST(torch.nn.Module):
    def __init__(
        self, D: int, S: int, stride: int, C_max: int, L_max: int,
        num_layers: int, nhead: int, dim_feedforward: int, out_dim: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = Tokenizer(S, stride)
        self.masking = Masking()
        self.embedding: torch.nn.Module = Embedding(D, S, C_max, L_max)
        self.transformer: torch.nn.Module = Transformer(
            D, num_layers, nhead, dim_feedforward
        )
        self.head_reconstruction: torch.nn.Module = HeadReconstruction(D, S)
        self.head_regression: torch.nn.Module = HeadRegression(D, out_dim)

    def forward(
        self, x: torch.Tensor,      # (B, C, T)
        channel_idx: torch.Tensor,  # (B, C)
        mask: bool = False, 
        task: Literal["regression", "reconstruction"] = "regression"
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        token = self.tokenizer.forward(x)   # (B, C, T) -> (B, C, L, S)
        if mask: 
            mlm_mask, mlm_mode, mlm_rand = self.masking(token)
        else:
            mlm_mask, mlm_mode, mlm_rand = None, None, None
        # (B, C, L, S) -> (B, C, L, D)
        x = self.embedding(token, channel_idx, mlm_mask, mlm_mode, mlm_rand)                           
        B, C, L, D = x.shape
        x = x.reshape((B, C*L, D))  # (B, C, L, D) -> (B, C*L, D)
        x = self.transformer(x)     # (B, C*L, D) -> (B, C*L, D)
        if task not in ["regression", "reconstruction"]:
            raise ValueError("task must be 'regression' or 'reconstruction'")
        elif task == "reconstruction":
            x = self.head_reconstruction(x)
            x = x.reshape((B, C, L, token.shape[-1]))
            if mask:
                # (B, C, L, S), (B, C, L, S), (B, C, L)
                return x, token, mlm_mask
            else:
                x = self.tokenizer.backward(x)
                token = self.tokenizer.backward(token)
                # (B, C, T), (B, C, T), None
                return x, token, None
        elif task == "regression":
            x = self.head_regression(x)
            # (B, out_dim), None, None
            return x, None, None