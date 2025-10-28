import torch


__all__ = [
    "HeadContrastive",
    "HeadReconstruction",
    "HeadRegression"
]


class HeadContrastive(torch.nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(D, 2*D),
            torch.nn.GELU(),
            torch.nn.Linear(2*D, D)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, D) -> (B, D)


class HeadReconstruction(torch.nn.Module):
    def __init__(self, D: int, S: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(D, S), 
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, C*L, D) -> (B, C*L, S)


class HeadRegression(torch.nn.Module):
    def __init__(self, D: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(D, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, D) -> (B, out_dim)


class HeadRegressionDeep(torch.nn.Module):
    def __init__(self, D: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(D, 2*D),
            torch.nn.GELU(),
            torch.nn.Linear(2*D, 2*D),
            torch.nn.GELU(),
            torch.nn.Linear(2*D, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, D) -> (B, out_dim)
