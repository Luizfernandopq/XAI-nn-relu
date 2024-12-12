import torch
import numpy as np
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, len_layers, list_len_neurons):
        super().__init__()
        self.len_layers = len_layers
        self.list_len_neurons = list_len_neurons

        self.layers = [ nn.Linear(self.list_len_neurons[layer],
                                  self.list_len_neurons[layer + 1]
                                  ) for layer in range(len_layers-1)]

        self.predictor = nn.Sequential(
            self.layers[0],
            nn.ReLU(),
            self.layers[1]
        )

    def forward(self, x):
        return self.predictor(x)

    # Intermediate layers only
    def get_layer_values(self, x, index_layer):
        if index_layer == self.len_layers - 1:
            raise ValueError("Apenas para camadas intermediárias")
        for index in range(index_layer):
            x = self.layers[index](x)
            x = torch.relu(x)
        return x
