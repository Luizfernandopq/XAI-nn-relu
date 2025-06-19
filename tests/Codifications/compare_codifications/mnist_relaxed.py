import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.back_explainer.milp.explainer import generate_explanation
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset
from src.legacy.codify_network import codify_network
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

if __name__ == '__main__':

    layers = [28*28, 32, 32, 10]

    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='../../NeuralNetworks/generate_network/data', train=True, download=True,
                              transform=transform)
    testset = datasets.MNIST(root='../../NeuralNetworks/generate_network/data', train=False, download=True,
                             transform=transform)

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

    mnist_network = ForwardReLU(layers)
    mnist_network.load_state_dict(torch.load(f'../../../Networks/mnist/Weights/mnist_net{layer_str}_weights01.pth',
                                             weights_only=True))


    start = time.perf_counter()
    mdl_relax, out_bounds_relax = relaxed_codify_network(mnist_network,
                                                         train_set.eat_other(test_set).to_dataframe(target=False),
                                                         relax_density=0.25)
    print("Time to codify relaxed:", time.perf_counter() - start)

    print(out_bounds_relax)

    start = time.perf_counter()
    mdl , out_bounds = codify_network(mnist_network,
                                      train_set.eat_other(test_set).to_dataframe(target=False))
    print("Time to codify:", time.perf_counter() - start)
    print(out_bounds)


