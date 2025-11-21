import torch


class Tokenizer:
    def __init__(self, segment_length: int, segment_stride: int) -> None:
        self.segment_length = segment_length
        self.segment_stride = segment_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # segment time dimension into overlapping tokens
        # unfold automatically drops the last segment if not enough length and 
        # still functions when there is nan in the input
        return x.unfold(    # (B, C, T) -> (B, C, L, S)
            dimension=2, size=self.segment_length, step=self.segment_stride
        ).contiguous()

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, S = x.shape
        T = (L - 1) * self.segment_stride + S
        y = (~torch.isnan(x)).to(x.dtype)
        x = torch.nan_to_num(x, nan=0.0)
        y = y.transpose(-1, -2).contiguous().view(B*C, S, L)
        x = x.transpose(-1, -2).contiguous().view(B*C, S, L)
        x = torch.nn.functional.fold(
            x,
            output_size=(T, 1), kernel_size=(S, 1),
            stride=(self.segment_stride, 1),
        )
        y = torch.nn.functional.fold(
            y,
            output_size=(T, 1), kernel_size=(S, 1),
            stride=(self.segment_stride, 1),
        )
        x = torch.where(
            y > 0, x / y, torch.full_like(x, float('nan'))
        ).reshape(B, C, T)
        return x
