import torch
import numpy as np
from torch import nn

"""
Uma rede neural com características específicas para o modelo de explicação denominada BackExplanation
"""
class ForwardReLU(nn.Module):
    def __init__(self, list_len_neurons):
        super().__init__()
        self.len_layers = len(list_len_neurons)
        self.list_len_neurons = list_len_neurons
        self.layers = []
        self.flatten = nn.Flatten()

        # N - 1 connections
        for layer in range(self.len_layers - 1):
            self.layers.append(nn.Linear(list_len_neurons[layer],
                                         list_len_neurons[layer+1]))

        self.list_layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = self.flatten(x)
        for layer in range(self.len_layers - 2):
            x = torch.relu(self.layers[layer](x))
        return self.layers[-1](x)

    # Intermediate layers only
    def get_layer_values(self, x, index_layer):
        if index_layer >= self.len_layers - 1 or index_layer <= 0:
            raise ValueError("Apenas para camadas intermediárias")
        for index in range(index_layer):
            x = torch.relu(self.layers[index](x))
        return x

    def get_all_neuron_values(self, x):
        values = [x.cpu().numpy()]
        for layer in range(self.len_layers - 2):
            x = torch.relu(self.layers[layer](x))
            values.append(x.detach().cpu().numpy())
        x = self.layers[-1](x)
        values.append(x.detach().cpu().numpy())
        return values