import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron
import torch
import numpy as np
from torch import nn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)
    
def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear

class HyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self,input_dim,output_dim):
        super(HyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_embedding_dim = 2
        self.hidden_size=128
        # Considers weight and bias parameters for generating adapter weights.
        self.weight_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim,self.hidden_size),
            nn.ReLU(),
            linear_layer(self.hidden_size,self.task_embedding_dim),
            )
        self.bias_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.hidden_size),
            nn.ReLU(),
            linear_layer(self.hidden_size,input_dim),
            )

    def forward(self,inputs):   ## input is two_dimension
        weight = self.weight_generator(inputs).view(self.input_dim, self.output_dim)
        inputs_bias=torch.mean(inputs,dim=0)
        #inputs_bias=F.normalize(inputs_bias)
        bias = self.bias_generator(inputs_bias).view(self.input_dim)     

        return weight,bias

class MMoEModel(torch.nn.Module):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.soft=nn.Softmax()
        self.expert_num = expert_num
        self.generate = HyperNet(expert_num,self.embed_output_dim)
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
     
    def Gate(self,inputs):
        parton_vector=np.random.default_rng().dirichlet((10, 5), 8704)
        parton_vector=torch.tensor(parton_vector).to(self.device).to(torch.float32)
        wei,bia=self.generate(parton_vector)
        self.task_num = 2
        output=[]
        #output= torch.nn.ModuleList([torch.nn.Sequential(F.linear(inputs,weight=wei,bias=bia).to(self.device), torch.nn.Softmax(dim=1)) for i in range(self.task_num)])
        for i in range(self.task_num):
            end=F.linear(inputs,weight=wei,bias=bia).to(self.device)
            end=self.soft(end)
            output.append(end)
        #output=[torch.nn.Sequential(F.linear(inputs,weight=wei,bias=bia).to(self.device), torch.nn.Softmax(dim=1)) for i in range(self.task_num)]
        return output
    
    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """

        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1) 
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = [self.Gate(emb)[i].unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results