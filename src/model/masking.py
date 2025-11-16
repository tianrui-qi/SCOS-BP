import torch


__all__ = ["Masking"]


class Masking:
    @staticmethod
    def masking(
        x: torch.Tensor,            # (B, C, L, S), float
        channel_idx: torch.Tensor,  # (B, C), long
        user_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
        user_src_key_padding_mask: (
            int | list[int] | tuple[int, ...] | torch.Tensor | None
        ) = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, L, _ = x.shape
        device = x.device

        """ 
        src_key_padding_mask
        (B, C, L), bool
        True = drop, False = keep
        """
        # init src_key_padding_mask as all False, i.e., keep all tokens
        src_key_padding_mask = torch.zeros(
            B, C, L, dtype=torch.bool, device=device
        )
        # drop tokens that has nan values
        src_key_padding_mask |= torch.isnan(x).any(dim=-1)
        # drop tokens from dropped channels
        src_key_padding_mask |= channel_idx.eq(-1).unsqueeze(-1).expand(B, C, L)
        # apply user_src_key_padding_mask
        if user_src_key_padding_mask is None:
            # no custom src_key_padding_mask
            pass
        elif isinstance(user_src_key_padding_mask, int):
            # a single channel index
            # drop all tokens of this channel for all samples
            src_key_padding_mask |= (
                channel_idx == user_src_key_padding_mask
            ).unsqueeze(-1).expand(B, C, L)
        elif (
            isinstance(user_src_key_padding_mask, (list, tuple)) and 
            all(isinstance(c, int) for c in user_src_key_padding_mask)
        ):
            # a list/tuple of channel indices
            # drop all tokens of these channels for all samples
            for c in user_src_key_padding_mask: src_key_padding_mask |= (
                channel_idx == c
            ).unsqueeze(-1).expand(B, C, L)
        elif (
            isinstance(user_src_key_padding_mask, torch.Tensor) and 
            user_src_key_padding_mask.shape == (C, L)
        ):
            # a (C, L) tensor specifying per-channel & per-token dropping
            # broadcast to (B, C, L) so every sample uses the same dropping
            src_key_padding_mask |= (
                user_src_key_padding_mask.bool().unsqueeze(0).expand(B, C, L)
            )
        else:
            raise ValueError

        """ 
        mask
        (B, C, L), long
        0 = no mask, 1 = mask (hide), 2 = mask (keep)
        """
        # init mask as all 0, i.e., no mask
        mask = torch.zeros(B, C, L, dtype=torch.long, device=device)
        # apply user_mask
        if user_mask is None:
            # no custom mask
            pass
        elif isinstance(user_mask, int):
            # a single channel index
            # mask (hide) all tokens of this channel for all samples
            mask[
                (channel_idx == user_mask).unsqueeze(-1).expand(B, C, L)
            ] = 1
        elif (
            isinstance(user_mask, (list, tuple)) and 
            all(isinstance(c, int) for c in user_mask)
        ):
            # a list/tuple of channel indices
            # mask (hide) all tokens of these channels for all samples
            for c in user_mask: mask[
                (channel_idx == c).unsqueeze(-1).expand(B, C, L)
            ] = 1
        elif (
            isinstance(user_mask, torch.Tensor) and 
            user_mask.shape == (C, L)
        ):
            # a (C, L) tensor specifying per-channel & per-token masking
            # broadcast to (B, C, L) so every sample uses the same mask
            # note that dtype can be long or bool
            # if bool, True will be converted to 1, i.e., mask (hide)
            # if long, directly use these values
            mask = user_mask.long().unsqueeze(0).expand(B, C, L).to(device)
        else:
            raise ValueError

        return src_key_padding_mask, mask   # (B, C, L), (B, C, L)

    @staticmethod
    def maskingContrastive_(
        x: torch.Tensor,            # (B, C, L, S), float
        channel_idx: torch.Tensor,  # (B, C), long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask, mask = Masking.masking(x, channel_idx)
        
        B, C, L = src_key_padding_mask.shape
        device = src_key_padding_mask.device

        # random drop channel for each sample, i.e., set channel_idx to -1
        # keep at least one valid channel for each sample
        # note that drop channel here is different from drop channel in 
        # dataset since contrastive learning needs diverse views
        valid = channel_idx.ne(-1)
        keep_num = (
            torch.rand(B, device=device) * valid.sum(dim=1).float()
        ).floor().long() + 1
        score = torch.rand(B, C, device=device)
        score = score.masked_fill(~valid, -float('inf'))
        sort_score, _ = torch.sort(score, dim=1, descending=True)
        thresh = sort_score.gather(
            1, (keep_num - 1).clamp(max=C-1).unsqueeze(1)
        )
        # in-place update channel_idx 
        keep = score >= thresh
        channel_idx[~keep] = -1
        # in-place drop tokens from dropped channels
        src_key_padding_mask |= channel_idx.eq(-1).unsqueeze(-1).expand(B, C, L)

        # keep 80%~100% of contiguous valid tokens for each channel
        valid = ~src_key_padding_mask
        idx = torch.arange(L, device=device).view(1, 1, L)
        first_idx = torch.where(
            valid, idx, torch.full((1, 1, 1), L, device=device)
        ).amin(dim=2)
        last_idx = torch.where(
            valid, idx, torch.full((1, 1, 1), -1, device=device)
        ).amax(dim=2)
        span_len = (last_idx - first_idx + 1).clamp(min=0)
        r = 0.8 + 0.2 * torch.rand(B, C, device=device)
        keep_len = torch.ceil(span_len.float() * r).long()
        keep_len = torch.minimum(keep_len, span_len)
        keep_len = torch.where(
            valid.any(dim=2), keep_len.clamp(min=1), 
            torch.zeros_like(keep_len)
        )
        delta_max = (span_len - keep_len).clamp(min=0)
        offset = torch.floor(
            torch.rand(B, C, device=device) * (delta_max + 1).float()
        ).long()
        start = first_idx + offset
        end = start + keep_len - 1
        keep_span = (idx >= start.unsqueeze(-1)) & (idx <= end.unsqueeze(-1))
        # in-place drop tokens outside the keep_span
        src_key_padding_mask |= ~keep_span

        # TODO： random drop individual tokens for each channel

        return src_key_padding_mask, mask   # (B, C, L), (B, C, L)

    @staticmethod
    def maskingReconstruction(
        x: torch.Tensor,            # (B, C, L, S), float
        channel_idx: torch.Tensor,  # (B, C), long
        p_point: float = 0.2, 
        p_span_small: tuple[float, float] = (0.0, 0.5),
        p_span_large: tuple[float, float] = (0.0, 1.0),
        p_hide: float = 0.9, p_keep: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask, mask = Masking.masking(x, channel_idx)

        B, C, L = src_key_padding_mask.shape
        device = src_key_padding_mask.device

        # src_key_padding_mask already set True for invalid tokens, i.e., nan
        # we only generate mask on valid tokens and then reconstruct them 
        # during training
        valid = ~src_key_padding_mask   # (B, C, L)
        # number of valid channels for each sample
        num_c = (channel_idx.ne(-1) & valid.any(dim=2)).sum(dim=1)  # (B,)

        # for each sample, randomly mask tokens with 0 to p_point_mask ratio
        r = torch.rand((B, C, L), device=device)
        rand_ratio = torch.rand(B, device=device) * p_point
        threshold = rand_ratio.view(B, 1, 1)
        hide = valid & (r < threshold)    # (B, C, L)
        # in case mask is all False, force at lease one True
        if not hide.any():
            all_valid = torch.nonzero(valid)
            pick = all_valid[
                torch.randint(len(all_valid), (1,), device=device)
            ][0]
            hide[*pick] = True
        # for each masked token, random select p_keep to keep
        r = torch.rand((B, C, L), device=device)
        keep = hide & (r < p_keep)
        # in-place update mask, 0 = not mask, 1 = mask (hide), 2 = mask (keep)
        mask[hide] = 1
        mask[keep] = 2

        # for each sample, randomly select one channel to generate span 
        # mask that mask p_span_min to p_span_max continuous tokens
        # only span on valid channels with at least one valid token
        span_c = channel_idx.ne(-1) & valid.any(dim=2)  # (B, C), bool
        # only span sample that has candidate channels
        span_s = span_c.any(dim=1).view(B, 1, 1)        # (B, 1, 1), bool
        # for each sample, randomly select one condidate channel
        score = torch.rand(B, C, device=device)
        score = score.masked_fill(~span_c, float("-inf"))
        span_c = score.argmax(dim=1)                    # (B,), long
        span_c = torch.nn.functional.one_hot(           # (B, C, 1), bool
            span_c, num_classes=C
        ).bool().unsqueeze(-1)
        # for each sample, calculate span mask positions
        # note that we generate span mask for all
        length_small = torch.randint(
            round(L * p_span_small[0]), round(L * p_span_small[1]) + 1,
            (B,), device=device
        )
        length_large = torch.randint(
            round(L * p_span_large[0]), round(L * p_span_large[1]) + 1,
            (B,), device=device
        )
        length = torch.where(num_c > 1, length_large, length_small)
        start = torch.floor(
            torch.rand(B, device=device) * (L - length).clamp(min=1)
        ).long()
        end = (start + length).clamp(max=L)
        pos = torch.arange(L, device=device).view(1, L)
        span_t = (pos >= start.view(-1, 1)) & (pos < end.view(-1, 1))
        span_t = span_t.unsqueeze(1).expand(-1, C, -1)  # (B, C, L), bool
        # combine
        # 1. valid token, i.e., ~src_key_padding_mask
        # 2. token selected to span
        # 3. channel selected to span
        # 4. sample selected to span and has candidate channels
        hide = valid & span_t & span_c & span_s         # (B, C, L)
        # for each sample, choose to hide or keep the span
        keep = hide & (
             torch.rand(B, device=device) < p_keep
        ).view(B, 1, 1).expand(-1, C, L)
        # in-place update mask, 0 = not mask, 1 = mask (hide), 2 = mask (keep)
        mask[hide] = 1
        mask[keep] = 2

        return src_key_padding_mask, mask   # (B, C, L), (B, C, L)


if __name__ == "__main__":
    x = torch.randn(2, 4, 10, 10)
    channel_idx = torch.tensor([[0, 1, -1, 3], [-1, -1, 2, -1]])
    src_key_padding_mask, mask = Masking.maskingContrastive_(x, channel_idx)
    print(channel_idx)  # test if channel_idx is changed in-place
    print(src_key_padding_mask)
    print(mask)
