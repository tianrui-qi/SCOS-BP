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


class HeadReconstructionCal(torch.nn.Module):
    def __init__(self, D: int, S: int) -> None:
        super().__init__()
        self.c_query = torch.nn.Parameter(torch.randn(D))
        self.c_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(D),
            torch.nn.Linear(D, 2 * D),
            torch.nn.GELU(),
            torch.nn.Linear(2 * D, 2 * D),
        )
        self.x_ln = torch.nn.LayerNorm(D)
        self.x_mlp = torch.nn.Sequential(
            torch.nn.Linear(D, D),
            torch.nn.GELU(),
            torch.nn.Linear(D, S),
        )
    def forward(
        self,
        x: torch.Tensor,            # (B, C*L, D)
        c: torch.Tensor,            # (B, P, D)
    ) -> torch.Tensor:
        att = (c * self.c_query).sum(-1) / (c.shape[-1] ** 0.5)
        weight = att.softmax(dim=-1)            # (B, P)
        c = (weight.unsqueeze(-1) * c).sum(1)   # (B, D)
        c = self.c_mlp(c)                       # (B, 2D)
        g, b = c.chunk(2, dim=-1)               # (B, D), (B, D)
        g = g.unsqueeze(1)                      # (B, 1, D)
        b = b.unsqueeze(1)                      # (B, 1, D)
        x = self.x_ln(x)                        # (B, N, D)
        x = g * x + b                           # (B, N, D)
        x = self.x_mlp(x)                       # (B, N, S)
        return x


class HeadReconstructionRaw(torch.nn.Module):
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
            torch.nn.Linear(D, 2*D),
            torch.nn.GELU(),
            torch.nn.Linear(2*D, 2*D),
            torch.nn.GELU(),
            torch.nn.Linear(2*D, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, D) -> (B, out_dim)
