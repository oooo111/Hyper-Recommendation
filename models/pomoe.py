import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
    
class HyperNet(nn.Module):
    """LeNet Hypernetwork
    """

    def __init__(self, ray_hidden_dim=64, out_dim=10,
                 target_hidden_dim=50, n_hidden=1, n_tasks=2, drop=0.05):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks
        self.dropout = nn.Dropout(drop)
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ray_hidden_dim)
        )
        
        setattr(self, f"gate_weights1", nn.Linear(ray_hidden_dim, target_hidden_dim * out_dim))
        setattr(self, f"gate_weights2", nn.Linear(ray_hidden_dim, out_dim * out_dim))
        
        setattr(self, f"gate_bias1", nn.Linear(ray_hidden_dim, out_dim))
        setattr(self, f"gate_bias2", nn.Linear(ray_hidden_dim, out_dim))
    def forward(self, ray):
        features = self.ray_mlp(ray)
        #features2=self.ray_mlp2(ray)
        out_dict = {}
        out_dict[f"gate_weights1"] = self.dropout(getattr(self, f"gate_weights1")(features))
        out_dict[f"gate_weights2"] = self.dropout(getattr(self, f"gate_weights2")(features))
        
        out_dict[f"gate_bias1"] = self.dropout(getattr(self, f"gate_bias1")(features).flatten())
        out_dict[f"gate_bias2"] = self.dropout(getattr(self, f"gate_bias2")(features).flatten())
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
            x, weight=weights[f'gate_weights1'].reshape(self.out_dim, self.target_hidden_dim),
            bias=weights[f'gate_bias1']
         )
        x = F.linear(x,weight=weights[f'gate_weights2'].reshape(self.out_dim, self.out_dim),bias=weights[f'gate_bias2'])
        x=F.relu(x)
        return x
    

class POMoEModel(torch.nn.Module):
    """
    A pytorch implementation of one-gate MoE Model.

    Reference:
        Jacobs, Robert A., et al. "Adaptive mixtures of local experts." Neural computation 3.1 (1991): 79-87.
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num,
                 expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.hypter = HyperNet(out_dim=expert_num, target_hidden_dim=self.embed_output_dim)
        self.expert = torch.nn.ModuleList(
            [MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in
             range(expert_num)])
        self.tower = torch.nn.ModuleList(
            [MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

        self.gate = TargetNet(target_hidden_dim=self.embed_output_dim, out_dim=expert_num)

    def forward(self, categorical_x, numerical_x, ray=None):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        #for name, parameter in self.hypter.named_parameters():
        #    print(name,parameter)
        #print(self.hypter.ray_mlp[0].state_dict())   
        weight = self.hypter(ray)
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = self.gate(x=emb, weights=weight).unsqueeze(1)
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1)
        # print('fea dim:', fea.shape, 'gate dim:', gate_value.shape)
        fea = torch.bmm(gate_value, fea).squeeze(1)

        results = [torch.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        return results