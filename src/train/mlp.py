import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, subset_dims, hidden_dim=512, output_dim=2):
        super().__init__()
        self.input_heads = nn.ModuleDict(
            {subset: nn.Linear(dim, hidden_dim) for subset, dim in subset_dims.items()}
        )
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, subset):
        h = self.input_heads[subset](x)
        return self.net(h)
