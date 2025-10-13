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
        self, D: int, S: int, stride: int, 
        C_max: int, L_max: int, dropout: float,
        num_layers: int, nhead: int, dim_feedforward: int, out_dim: int,
        freeze_embedding: bool = False, freeze_transformer: int = 0, **kwargs
    ) -> None:
        super().__init__()
        # modules
        self.tokenizer = Tokenizer(S, stride)
        self.masking = Masking()
        self.embedding: torch.nn.Module = Embedding(
            D, S, C_max, L_max, dropout
        )
        self.transformer: torch.nn.Module = Transformer(
            D, num_layers, nhead, dim_feedforward
        )
        self.head_reconstruction: torch.nn.Module = HeadReconstruction(D, S)
        self.head_regression: torch.nn.Module = HeadRegression(D, out_dim)
        # freeze
        self._freeze(freeze_embedding, freeze_transformer)

    def _freeze(self, freeze_embedding: bool, freeze_transformer: int) -> None:
        if freeze_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
        if freeze_transformer > 0:
            for i, layer in enumerate(self.transformer.encoder.layers):
                if i < freeze_transformer:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self, x: torch.Tensor,      # (B, C, T)
        channel_idx: torch.Tensor,  # (B, C)
        mask: bool = False, 
        task: Literal["regression", "reconstruction"] = "regression"
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
                # (num of masked tokens, S), (num of masked tokens, S)
                return x[mlm_mask], token[mlm_mask].detach()
            else:
                x = self.tokenizer.backward(x)
                token = self.tokenizer.backward(token)
                # (B, C, T), (B, C, T)
                return x, token.detach()
        elif task == "regression":
            x = self.head_regression(x)
            # (B, out_dim), None
            return x, None