import torch


__all__ = []


class Embedding(torch.nn.Module):
    def __init__(
        self, D: int, S: int, C_max: int, L_max: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embd_token = EmbeddingToken(D, S)
        self.embd_channel = EmbeddingChannel(D, C_max)
        self.embd_position = EmbeddingPosition(D, L_max)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self, 
        x: torch.Tensor,                        # (B, C, L, S)
        channel_idx: torch.Tensor,              # (B, C), long
        src_key_padding_mask: torch.Tensor,     # (B, C, L), bool
        mask: torch.Tensor,                     # (B, C, L), long
    ) -> torch.Tensor:
        x = self.dropout(
            self.embd_token(x, src_key_padding_mask, mask) +
            self.embd_channel(channel_idx) + 
            self.embd_position(x.shape[2])
        )
        return x    # (B, C, L, D)


class EmbeddingToken(torch.nn.Module):
    def __init__(self, D: int, S: int) -> None:
        super().__init__()
        # linear layer to project token to D dimension embedding
        self.te = torch.nn.Linear(S, D)
        # learnable embedding for token with mask
        self.mask_emb = torch.nn.Parameter(torch.zeros(1, 1, 1, D))
        torch.nn.init.trunc_normal_(self.mask_emb, std=0.02)

    def forward(
        self, 
        x: torch.Tensor,                        # (B, C, L, S), float
        src_key_padding_mask: torch.Tensor,     # (B, C, L), bool
        mask: torch.Tensor,                     # (B, C, L), long
    ) -> torch.Tensor:
        # replace tokens that true in src_key_padding_mask to zero
        # note that we already drop tokens that has nan values and update
        # src_key_padding_mask accordingly before calling
        x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        # project each token to D dimension embedding
        # if a token contains nan, torch.nn.Linear will output nan for
        # that token's embedding
        x = self.te(x)  # (B, C, L, S) -> (B, C, L, D)
        # replace tokens that true in src_key_padding_mask to zero
        # repeat since linear layer has bias term
        x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        # replace tokens that are 1 in mask, i.e., hide, to mask_emb
        # 0 = not mask, 1 = mask (hide), 2 = mask (keep)
        x = torch.where(
            mask.unsqueeze(-1) == 1, self.mask_emb.expand_as(x), x
        )
        return x    # (B, C, L, D)


class EmbeddingChannel(torch.nn.Module):
    def __init__(self, D: int, C_max: int) -> None:
        super().__init__()
        self.ce = torch.nn.Embedding(
            num_embeddings=C_max+1, embedding_dim=D, padding_idx=0
        )

    def forward(self, channel_idx: torch.Tensor) -> torch.Tensor:
        x = self.ce(channel_idx + 1)    # (B, C) -> (B, C, D)
        x = x.unsqueeze(2)              # (B, C, D) -> (B, C, 1, D)
        return x                        # (B, C, 1, D)


class EmbeddingPosition(torch.nn.Module):
    pe: torch.Tensor

    def __init__(self, D: int, L_max: int) -> None:
        super().__init__()
        position = torch.arange(L_max).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, D, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / D)
        ) 
        pe = torch.zeros(L_max, D)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[None, None, ...]        # (1, 1, L_max, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, L: int) -> torch.Tensor:
        x = self.pe[:, :, :L, :]
        return x    # (1, 1, L, D)
