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
        self, x: torch.Tensor,                  # (B, C, L, S)
        channel_idx: torch.Tensor,              # (B, C)
        mlm_mask: None | torch.Tensor = None,   # (B, C, L)
        mlm_mode: None | torch.Tensor = None,   # (B, C, L)
        mlm_rand: None | torch.Tensor = None,   # (B, C, L, 3)
    ) -> torch.Tensor:
        x = self.dropout(
            self.embd_token(x, mlm_mask, mlm_mode, mlm_rand) + 
            self.embd_channel(channel_idx) + 
            self.embd_position(x.shape[2])
        )
        return x    # (B, C, L, D)


class EmbeddingToken(torch.nn.Module):
    def __init__(self, D: int, S: int) -> None:
        super().__init__()
        # linear layer to project token to D dimension embedding
        self.te = torch.nn.Linear(S, D)
        # learnable embedding for token with nan
        self.nan_emb = torch.nn.Parameter(torch.zeros(1, 1, 1, D))
        torch.nn.init.trunc_normal_(self.nan_emb, std=0.02)
        # learnable embedding for token with mask
        self.mask_emb = torch.nn.Parameter(torch.zeros(1, 1, 1, D))
        torch.nn.init.trunc_normal_(self.mask_emb, std=0.02)

    def forward(
        self, x: torch.Tensor,                  # (B, C, L, S)
        mlm_mask: None | torch.Tensor = None,   # (B, C, L)
        mlm_mode: None | torch.Tensor = None,   # (B, C, L)
        mlm_rand: None | torch.Tensor = None,   # (B, C, L, 3)
    ) -> torch.Tensor:
        # 1. project each token to D dimension embedding
        # if a token contains nan, torch.nn.Linear will output nan for
        # that token's embedding
        x = self.te(x)      # (B, C, L, S) -> (B, C, L, D)
        # 2. replace embedding of token with nan to learnable embedding
        x = torch.where(    # (B, C, L, D)
            torch.isnan(x).any(dim=-1, keepdim=True),
            self.nan_emb.expand_as(x), x,
        )

        if mlm_mask is None or mlm_mode is None or mlm_rand is None or \
        self.training is False:
            return x        # (B, C, L, D)

        # make a clone before applying masking since when applying random
        # replacement we need the original embedding
        x_clone = x.clone()

        # 3. apply masking if provided
        # mlm_mask: (B, C, L), mlm_mode: (B, C, L)
        mask = mlm_mask & (mlm_mode == 1)
        x = torch.where(
            mask.unsqueeze(-1), self.mask_emb.expand_as(x), x
        )
        # 4. apply random replacement if provided
        rand_pos = (mlm_mask & (mlm_mode == 2))     # (B, C, L)
        if rand_pos.any():
            tgt_idx = rand_pos.nonzero()
            src_idx = mlm_rand[
                tgt_idx[:, 0], tgt_idx[:, 1], tgt_idx[:, 2]
            ].long()
            b_t, c_t, l_t = tgt_idx.unbind(dim=1)
            b_s, c_s, l_s = src_idx.unbind(dim=1)
            x[b_t, c_t, l_t] = x_clone[b_s, c_s, l_s]

        return x            # (B, C, L, D)


class EmbeddingChannel(torch.nn.Module):
    def __init__(self, D: int, C_max: int) -> None:
        super().__init__()
        self.ce = torch.nn.Embedding(num_embeddings=C_max, embedding_dim=D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ce(x)      # (B, C) -> (B, C, D)
        x = x.unsqueeze(2)  # (B, C, D) -> (B, C, 1, D)
        return x            # (B, C, 1, D)


class EmbeddingPosition(torch.nn.Module):
    pe: torch.Tensor

    def __init__(self, D: int, L_max: int) -> None:
        super().__init__()
        position = torch.arange(L_max).unsqueeze(1).float()     # (L_max, 1)
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
        return self.pe[:, :, :L, :]     # (1, 1, L, D)