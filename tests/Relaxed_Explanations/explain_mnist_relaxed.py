from time import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset
from src.legacy.codify_network import codify_network
from src.legacy.explication import get_miminal_explanation
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

def run(layers, relaxation=0.25):

    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='../NeuralNetworks/generate_network/data', train=True, download=True,
                              transform=transform)
    testset = datasets.MNIST(root='../NeuralNetworks/generate_network/data', train=False, download=True,
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
    mnist_network.load_state_dict(torch.load(f'../../Networks/mnist/Weights/mnist_net{layer_str}_weights01.pth',
                                             weights_only=True))

    all_set = train_set.eat_other(test_set)
    df = all_set.to_dataframe()

    relaxed_model, relaxed_bounds = relaxed_codify_network(mnist_network, all_set.to_dataframe(target=False),
                                                           relax_density=relaxation)

    len_inputs = []
    start = time()
    for index, instance in df[:100].iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")
        # print(instance)
        inputs = get_miminal_explanation(relaxed_model, instance, target, relaxed_bounds, 10)
        len_inputs.append(len(inputs))
        # print(inputs, "\n", len_inputs)

    print(f"Tempo Relaxed: {time() - start}")
    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")

    # ------------------------------------ PART 2 ----------------------------------------

    model, bounds_legacy = codify_network(mnist_network, all_set.to_dataframe(target=False))

    len_inputs = []

    start = time()
    for index, instance in df[:100].iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")
        # print(instance)
        inputs = get_miminal_explanation(model, instance, target, bounds_legacy, 10)
        len_inputs.append(len(inputs))
        # print(inputs, "\n", len_inputs)
    print(f"Tempo legacy: {time() - start}")

    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")

if __name__ == '__main__':
    list_layers = [[28 * 28, 16, 10],
                   [28 * 28, 16, 16, 10],
                   [28 * 28, 32, 10],
                   [28 * 28, 32, 32, 10],
                   [28 * 28, 64, 10],
                   [28 * 28, 64, 64, 10]]
    for layers in list_layers[:-2]:
        print(f"Rodando: {layers}")
        run(layers)
