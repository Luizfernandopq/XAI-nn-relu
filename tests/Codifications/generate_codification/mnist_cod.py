import time
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from src.legacy.codify_network import codify_network
from src.legacy.explication import get_miminal_explanation
from src.modeler.explainer import generate_explanation
from src.modeler.milp.Codificator import Codificator
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer
from src.modeler.network.SimpleDataset import SimpleDataset


def run(layers):
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Configurações
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='../../NeuralNetworks/generate_network/data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='../../NeuralNetworks/generate_network/data', train=False, download=True, transform=transform)

    X_train = trainset.data.numpy()

    y_train = trainset.targets.numpy()
    X_train = X_train.reshape(X_train.shape[0], -1)

    X_test = testset.data.numpy()

    y_test = testset.targets.numpy()
    X_test = X_test.reshape(X_test.shape[0], -1)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)

    # Modelo, função de perda e otimizador
    mnist_net = ForwardReLU(list_len_neurons=layers)
    mnist_net.load_state_dict(torch.load(f'../../../Networks/mnist/Weights/mnist_net{layer_str}_weights01.pth',
                                            weights_only=True))

    mnist_net.eval()

    # weights = [layer.weight.detach().numpy() for layer in mnist_net.layers if hasattr(layer, 'weight')]
    # biases = [layer.bias.detach().numpy() for layer in mnist_net.layers if
    #           hasattr(layer, 'bias') and layer.bias is not None]

    codificator = Codificator(mnist_net, train_set.eat_other(test_set).to_dataframe(target=False))
    bounds = codificator.codify_network_find_bounds()

    codificator.milp_represetation.export_as_sav(f"../../../Networks/mnist/Milps/mnist_milp{layer_str}.sav")

    with open(f"../../../Networks/mnist/Bounds/mnist_bound{layer_str}.pkl", "wb") as f:
        pickle.dump(bounds, f)



if __name__ == '__main__':
    list_layers = [[28*28, 16, 10],
                   [28*28, 16, 16, 10],
                   [28*28, 32, 10],
                   [28*28, 32, 32, 10],
                   [28*28, 64, 10],
                   [28*28, 64, 64, 10]]

    for layers in list_layers:
        print(f"Runnig codificator for: {layers}")
        run(layers)

