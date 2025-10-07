import torch


__all__ = []


class Tokenizer:
    def __init__(self, segment_length: int, segment_stride: int) -> None:
        self.segment_length = segment_length
        self.segment_stride = segment_stride

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # segment time dimension into overlapping tokens
        # unfold automatically drops the last segment if not enough length and 
        # still functions when there is nan in the input
        return x.unfold(    # (B, C, T) -> (B, C, L, S)
            dimension=2, size=self.segment_length, step=self.segment_stride
        ).contiguous()