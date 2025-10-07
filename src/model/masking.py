import torch


__all__ = []


class Masking:
    def __init__(
        self, p_mlm: float = 0.15,
        p_keep: float = 0.1, p_mask: float = 0.8, p_rand: float = 0.1, 
    ) -> None:
        self.p_mlm  = p_mlm
        self.p_keep = p_keep
        self.p_mask = p_mask
        self.p_rand = p_rand

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, L, _ = x.shape    # (B, C, L, S)
        device = x.device

        # mlm_mask, (B, C, L)
        r = torch.rand((B, C, L), device=device)
        valid = ~torch.isnan(x).any(dim=-1)     # (B, C, L)
        mlm_mask = (r < self.p_mlm) & valid
        # in case mlm_mask is all False, force at lease one True
        if not mlm_mask.any():
            all_valid = torch.nonzero(valid)    # (N, 3)
            pick = all_valid[
                torch.randint(len(all_valid), (1,), device=device)
            ][0]
            mlm_mask[*pick] = True

        # mlm_mode, (B, C, L), 0=keep, 1=mask, 2=random
        r = torch.rand((B, C, L), device=device)
        mlm_mode = torch.zeros((B, C, L), dtype=torch.int, device=device)
        mlm_mode[mlm_mask & (r < self.p_mask)] = 1
        mlm_mode[mlm_mask & (r >= 1 - self.p_rand)] = 2

        # rand_src, (B, C, L, 3)
        mlm_rand = torch.zeros((B, C, L, 3), dtype=torch.int, device=device)
        for pos in torch.nonzero((mlm_mode == 2)):
            # choose from same channel that is valid, i.e., not nan
            # no need to exclude self so that we must have true value to sample
            all_valid = torch.nonzero((valid[:, pos[1]] == True)).squeeze(1)
            pick = all_valid[torch.randint(len(all_valid), (1,))][0]
            mlm_rand[*pos] = torch.tensor([pick[0], pos[1], pick[1]])

        return mlm_mask, mlm_mode, mlm_rand