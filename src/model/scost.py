import torch

from .tokenizer import Tokenizer
from .masking import Masking
from .embedding import Embedding
from .transformer import Transformer
from .head import (
    HeadContrastive, 
    HeadReconstructionCal,
    HeadReconstructionRaw, 
    HeadRegression
)


__all__ = ["SCOST"]


class SCOST(torch.nn.Module):
    def __init__(
        self, D: int, 
        S: int, stride: int, 
        C_max: int, L_max: int,
        num_layers: int, nhead: int, dim_feedforward: int,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        # modules
        self.tokenizer = Tokenizer(segment_length=S, segment_stride=stride)
        self.embedding = Embedding(D, S, C_max, L_max)
        self.transformer = Transformer(D, num_layers, nhead, dim_feedforward)
        # head
        self.head_contrastive = HeadContrastive(D)
        self.head_recon_cal = HeadReconstructionCal(D, S)
        self.head_recon_raw = HeadReconstructionRaw(D, S)
        self.head_regression = HeadRegression(D, out_dim=out_dim)

    def freeze(
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
        x: torch.Tensor,                # (B, C, T), float
        x_channel_idx: torch.Tensor,    # (B, C), long
        user_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        user_src_key_padding_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        pool: bool = True,
    ) -> torch.Tensor:
        x = self.tokenizer.forward(x)   # (B, C, L, S)
        src_key_padding_mask, mask = (  # (B, C, L), (B, C, L)
            Masking.masking(x, x_channel_idx, 
                user_mask=user_mask,
                user_src_key_padding_mask=user_src_key_padding_mask
            )
        )
        x = self.embedding(             # (B, C, L, D)
            x, x_channel_idx, src_key_padding_mask, mask
        )
        x = self.transformer(           # (B, D) or (B, C*L, D)
            x.reshape(x.shape[0], -1, x.shape[-1]),
            src_key_padding_mask.reshape(x.shape[0], -1),
            pool=pool,
        )
        return x                        # (B, D) or (B, C*L, D)

    def forwardContrastive(
        self, 
        x: torch.Tensor,                # (B, C, T), float
        x_channel_idx: torch.Tensor,    # (B, C), long
    ) -> torch.Tensor:
        x = self.tokenizer.forward(x)   # (B, C, L, S)
        src_key_padding_mask, mask = (  # (B, C, L), (B, C, L)
            Masking.maskingContrastive_(x, x_channel_idx)
        )
        x = self.embedding(             # (B, C, L, D)
            x, x_channel_idx, src_key_padding_mask, mask
        )
        x = self.transformer(           # (B, D)
            x.reshape(x.shape[0], -1, x.shape[-1]),
            src_key_padding_mask.reshape(x.shape[0], -1),
        )
        x = self.head_contrastive(x)    # (B, D)
        return x

    def forwardReconstructionCal(
        self, 
        x: torch.Tensor,                # (B, C, T), float
        x_channel_idx: torch.Tensor,    # (B, C), long
        c: torch.Tensor,                # (B, P, C, T), float
        c_channel_idx: torch.Tensor,    # (B, P, C), long
        y: torch.Tensor | None = None,  # (B, C, T), float
        user_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        user_src_key_padding_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        p_point: float = 0.2, 
        p_span_small: tuple[float, float] = (0.0, 0.5),
        p_span_large: tuple[float, float] = (0.0, 1.0),
        p_hide: float = 0.9, p_keep: float = 0.1,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.tokenizer.forward(x)   # (B, C, L, S)
        y = (                           # (B, C, L, S)
            x.detach() if y is None else self.tokenizer.forward(y).detach()
        )
        src_key_padding_mask, mask = (  # (B, C, L), (B, C, L)
            Masking.masking(x, x_channel_idx,
                user_mask=user_mask,
                user_src_key_padding_mask=user_src_key_padding_mask
            )
        ) if user_mask is not None else (  
            Masking.maskingReconstruction(x, x_channel_idx,
                p_point=p_point, 
                p_span_small=p_span_small, 
                p_span_large=p_span_large, 
                p_hide=p_hide,
                p_keep=p_keep,
            ) 
        )
        x = self.embedding(             # (B, C, L, D)
            x, x_channel_idx, src_key_padding_mask, mask
        )
        x = self.transformer(           # (B, C*L, D)
            x.reshape(x.shape[0], -1, x.shape[-1]),
            src_key_padding_mask.reshape(x.shape[0], -1),
            pool=False,
        )
        c = self.forward(               # (B, P, D)
            c.reshape(-1, c.shape[-2], c.shape[-1]), 
            c_channel_idx.reshape(-1, c_channel_idx.shape[-1]),
        ).reshape(c.shape[0], c.shape[1], -1)
        x = self.head_recon_cal(x, c)   # (B, C*L, S)
        return (
            # return waveform for all tokens by inverse tokenizer
            (                           # B, C, T)
                self.tokenizer.backward(x.reshape_as(y)),
                self.tokenizer.backward(y),
            )
        ) if user_mask is not None else (
            # return waveform at masked token only
            (                           # #mask, S), (#mask, S)
                x.reshape_as(y)[mask != 0],
                y[mask != 0],
            )
        )

    def forwardReconstructionRaw(
        self, 
        x: torch.Tensor,                # (B, C, T), float
        x_channel_idx: torch.Tensor,    # (B, C), long
        user_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        user_src_key_padding_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        p_point: float = 0.2, 
        p_span_small: tuple[float, float] = (0.0, 0.5),
        p_span_large: tuple[float, float] = (0.0, 1.0),
        p_hide: float = 0.9, p_keep: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokenizer.forward(x)   # (B, C, L, S)
        y = x.detach()                  # (B, C, L, S)
        src_key_padding_mask, mask = (  # (B, C, L), (B, C, L)
            Masking.masking(x, x_channel_idx,
                user_mask=user_mask,
                user_src_key_padding_mask=user_src_key_padding_mask
            )
        ) if user_mask is not None else (  
            Masking.maskingReconstruction(x, x_channel_idx,
                p_point=p_point, 
                p_span_small=p_span_small, 
                p_span_large=p_span_large, 
                p_hide=p_hide,
                p_keep=p_keep,
            ) 
        )
        x = self.embedding(             # (B, C, L, D)
            x, x_channel_idx, src_key_padding_mask, mask
        )
        x = self.transformer(           # (B, C*L, D)
            x.reshape(x.shape[0], -1, x.shape[-1]),
            src_key_padding_mask.reshape(x.shape[0], -1),
            pool=False,
        )
        x = self.head_recon_raw(x)      # (B, C*L, S)
        return (
            # return waveform for all tokens by inverse tokenizer
            (                           # B, C, T), (B, C, T)
                self.tokenizer.backward(x.reshape_as(y)),
                self.tokenizer.backward(y),
            )
        ) if user_mask is not None else (
            # return waveform at masked token only
            (                           # #mask, S), (#mask, S)
                x.reshape_as(y)[mask != 0],
                y[mask != 0],
            )
        )

    def forwardRegression(
        self, 
        x: torch.Tensor,                # (B, C, T), float
        x_channel_idx: torch.Tensor,    # (B, C), long
        user_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        user_src_key_padding_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
    ) -> torch.Tensor:
        x = self.forward(               # (B, D)
            x, x_channel_idx,
            user_mask=user_mask,
            user_src_key_padding_mask=user_src_key_padding_mask,
        )
        x = self.head_regression(x)     # (B, out_dim)
        return x
