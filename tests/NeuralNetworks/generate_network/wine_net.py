from time import time

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.modeler.explainer import generate_explanation
from src.modeler.milp.Codificator import Codificator
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer
from src.modeler.network.SimpleDataset import SimpleDataset


if __name__ == '__main__':

    # Data
    bunch = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target,
                                                        test_size=0.33,
                                                        random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)

    # Network and Train

    layers = [13, 16, 16, 3]
    wine_network = ForwardReLU(layers)
    trainer = ForwardReluTrainer(wine_network, train_loader=None)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(400)

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]
    print(weights)
    i = int(input("Network number: "))
    torch.save(wine_network.state_dict(), f'../../../Networks/N-Weights/wine_net_13x16x16x3_weights{i:02}.pth')
