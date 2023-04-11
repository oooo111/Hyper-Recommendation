from torch import nn
import torch
import torch.nn.functional as F

class HyperNet(nn.Module):
    """LeNet Hypernetwork
    """

    def __init__(self, ray_hidden_dim=100, out_dim=10,
                 target_hidden_dim=50, n_hidden=1, n_tasks=2, drop=0.05):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks
        self.dropout = nn.Dropout(drop)
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim)
        )

        setattr(self, f"gate_weights", nn.Linear(ray_hidden_dim, target_hidden_dim * out_dim))
        setattr(self, f"gate_bias", nn.Linear(ray_hidden_dim, out_dim))

    def forward(self, ray):
        features = self.ray_mlp(ray)

        out_dict = {}
        out_dict[f"gate_weights"] = self.dropout(getattr(self, f"gate_weights")(features))
        out_dict[f"gate_bias"] = self.dropout(getattr(self, f"gate_bias")(features).flatten())

        return out_dict


class TargetNet(nn.Module):
    """LeNet target network
    """
    def __init__(self, out_dim=10, target_hidden_dim=50):
        super().__init__()
        self.out_dim = out_dim
        self.target_hidden_dim = target_hidden_dim
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, weights=None):
        x = F.linear(
            x, weight=weights[f'gate_weights'].reshape(self.out_dim, self.target_hidden_dim),
            bias=weights[f'gate_bias']
         )
        return self.softmax(x)
