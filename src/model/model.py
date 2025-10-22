import torch

from typing import Literal

from .tokenizer import Tokenizer
from .masking import Masking
from .embedding import Embedding
from .transformer import Transformer
from .head import HeadContrastive, HeadReconstruction, HeadRegression


__all__ = ["SCOST"]


class SCOST(torch.nn.Module):
    def __init__(
        self, D: int, S: int, stride: int, 
        C_max: int, L_max: int,
        num_layers: int, nhead: int, dim_feedforward: int,
        out_dim: int = 2,
        freeze_embedding: bool = False, freeze_transformer: int = 0, 
        **kwargs
    ) -> None:
        super().__init__()
        # modules
        self.tokenizer = Tokenizer(S, stride)
        self.masking = Masking()
        self.embedding = Embedding(D, S, C_max, L_max)
        self.transformer = Transformer(D, num_layers, nhead, dim_feedforward)
        # head
        self.head_contrastive = HeadContrastive(D)
        self.head_reconstruction = HeadReconstruction(D, S)
        self.head_regression = HeadRegression(D, out_dim=out_dim)
        # freeze
        self._freeze(freeze_embedding, freeze_transformer)

    def _freeze(
        self, freeze_embedding: bool = False, freeze_transformer: int = 0,
    ) -> None:
        if freeze_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
        if freeze_transformer > 0:
            for i, layer in enumerate(self.transformer.encoder.layers):
                if i < freeze_transformer:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self, 
        x: torch.Tensor,            # (B, C, T), float
        channel_idx: torch.Tensor,  # (B, C), long
        masking_type: Literal["contrastive", "reconstruction"] | None = None,
        pool: bool = True,
        head_type: Literal[
            "contrastive", "reconstruction", "regression"
        ] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # override pool if head_type is given
        if head_type == "contrastive":    pool = True
        if head_type == "reconstruction": pool = False
        if head_type == "regression":     pool = True
        # tokenizing
        x = self.tokenizer.forward(x)   # (B, C, L, S)
        y = x.detach()                  # (B, C, L, S)
        # masking
        src_key_padding_mask, mask = self.masking(  # (B, C, L), (B, C, L)
            x, channel_idx, masking_type=masking_type
        )
        # embedding
        x = self.embedding(     # (B, C, L, D)
            x, channel_idx, src_key_padding_mask, mask
        )
        # transformer
        x = self.transformer(   # (B, C*L, D) or (B, D)
            x.reshape(x.shape[0], -1, x.shape[-1]),
            src_key_padding_mask.reshape(x.shape[0], -1),
            pool=pool
        )
        # head
        if head_type is None:
            return x, None  # (B, C*L, D) or (B, D), None
        if head_type == "contrastive":
            x = self.head_contrastive(x)
            return x, None  # (B, D), None
        if head_type == "reconstruction":
            x = self.head_reconstruction(x)     # (B, C*L, S)
            x = x.reshape_as(y)                 # (B, C, L, S)
            if masking_type == "reconstruction":
                x = x[mask != 0]
                y = y[mask != 0]
            if masking_type != "reconstruction":
                x = self.tokenizer.backward(x)  # (B, C, T)
                y = self.tokenizer.backward(y)  # (B, C, T)
            return x, y     # (#mask, S), (#mask, S) or (B, C, T), (B, C, T)
        if head_type == "regression":
            x = self.head_regression(x)
            return x, None  # (B, out_dim), None
