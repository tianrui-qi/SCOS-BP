import torch


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
    def __init__(self, D: int, S: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(D),
            torch.nn.Linear(D, D),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(D, S)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (..., D) -> (..., S)


class HeadAdapter(torch.nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1, D))
        self.beta  = torch.nn.Parameter(torch.zeros(1, 1, D))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1 + self.gamma) + self.beta     # (B, L, D) -> (B, L, D)
