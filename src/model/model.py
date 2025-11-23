import torch

from .tokenizer import Tokenizer
from .masking import Masking
from .embedding import Embedding
from .transformer import Transformer
from .head import (
    HeadAdapter,
    HeadContrastive, 
    HeadReconstruction, 
    HeadRegression
)


class SCOST(torch.nn.Module):
    def __init__(
        self, D: int, 
        S: int, stride: int, 
        C_max: int, L_max: int,
        num_layers: int, nhead: int, dim_feedforward: int,
    ) -> None:
        super().__init__()
        # backbone
        self.tokenizer = Tokenizer(segment_length=S, segment_stride=stride)
        self.embedding = Embedding(D, S, C_max, L_max)
        self.transformer = Transformer(D, num_layers, nhead, dim_feedforward)
        # adapter for subject-specific finetune
        self.head_adapter = HeadAdapter(D)
        # heads for different pretext tasks
        self.head_contrastive = HeadContrastive(D)
        self.head_reconstruction = HeadReconstruction(D, S)
        self.head_regression = HeadRegression(D, S)

    def freeze(
        self, 
        freeze_embedding: bool = False, 
        freeze_transformer: int = 0,
        freeze_head: bool = False,
    ) -> None:
        if freeze_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
        if freeze_transformer > 0:
            for i, layer in enumerate(self.transformer.encoder.layers):
                if i < freeze_transformer:
                    for param in layer.parameters():
                        param.requires_grad = False
        if freeze_head:
            for param in self.head_contrastive.parameters():
                    param.requires_grad = False
            for param in self.head_reconstruction.parameters():
                    param.requires_grad = False
            for param in self.head_regression.parameters():
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

    def forwardReconstruction(
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
        x = self.head_reconstruction(x) # (B, C*L, S)
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
        adapter: bool = False,
    ) -> torch.Tensor:
        B, C, T = x.shape
        x = self.forward(               # (B, C*L, D)
            x, x_channel_idx,
            user_mask=user_mask,
            user_src_key_padding_mask=user_src_key_padding_mask,
            pool=False,
        )
        x = x.reshape(B, C, -1, x.shape[-1])    # (B, C, L, D)
        x = x.mean(1)                   # (B, L, D)
        if adapter: 
            x = self.head_adapter(x)    # (B, L, D)
        x = self.head_regression(x)     # (B, L, S)
        x = x.unsqueeze(1)              # (B, 1, L, S)
        x = self.tokenizer.backward(x)  # (B, 1, T)
        x = x.squeeze(1)                # (B, T)
        return x
