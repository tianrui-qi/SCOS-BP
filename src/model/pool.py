import torch


class Pool:
    @staticmethod
    def PoolMean(
        x: torch.Tensor,                # (B, C, L, D), float
        src_key_padding_mask: (         # (B, C, L), bool
            torch.Tensor | None
        ) = None,
        pool_dim: int | tuple[int, ...] | None = (1, 2),
    ) -> torch.Tensor:
        # if no pooling, return x directly
        if pool_dim is None: return x
        # if no mask, do normal mean pooling
        if src_key_padding_mask is None: return x.mean(dim=pool_dim)
        # unify dim format
        if isinstance(pool_dim, int): pool_dim = (pool_dim,)
        # 1 = keep, 0 = drop
        keep = (~src_key_padding_mask).unsqueeze(-1).to(x.dtype)
        num = (x * keep).sum(dim=pool_dim)
        denom = keep.sum(dim=pool_dim).clamp_min(1.0)
        x = num / denom
        return x
