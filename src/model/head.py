import torch


__all__ = []


class HeadReconstruction(torch.nn.Module):
    def __init__(self, D: int, S: int) -> None:
        super().__init__()
        self.rh = torch.nn.Sequential(
            torch.nn.LayerNorm(D), 
            torch.nn.Linear(D, 2*D), 
            torch.nn.GELU(), 
            torch.nn.Linear(2*D, S)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rh(x)   # (B, C*L, D) -> (B, C*L, S)


class HeadRegression(torch.nn.Module):
    def __init__(self, D: int, out_dim=2) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(D), 
            torch.nn.Linear(D, 2*D), 
            torch.nn.GELU(), 
            torch.nn.Dropout(0.1), 
            torch.nn.Linear(2*D, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)   # (B, C*L, D) -> (B, D)
        return self.mlp(x)  # (B, D) -> (B, out_dim)