import torch
from torch import nn

class MnistNN(nn.Module):
    def __init__(self, len_layers, list_len_neurons):
        super().__init__()
        self.len_layers = len_layers
        self.layers = []
        self.list_len_neurons = list_len_neurons
        for i in range(len(list_len_neurons) - 1):
            self.layers.append(nn.Linear(list_len_neurons[i], list_len_neurons[i + 1]))
            self.layers.append(nn.ReLU())

        self.predictor = nn.Sequential(nn.Flatten(), *self.layers)

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
