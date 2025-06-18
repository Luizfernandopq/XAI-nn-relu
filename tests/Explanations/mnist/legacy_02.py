import time

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
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer
from src.modeler.network.SimpleDataset import SimpleDataset


def run():
    # Configurações
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    # Modelo, função de perda e otimizador
    model = ForwardReLU(list_len_neurons=[28 * 28, 64, 10])

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = ForwardReluTrainer(model=model,
                                 train_loader=train_loader,
                                 device=device,
                                 optimizer=optimizer,
                                 test_loader=test_loader)

    trainer.fit(epochs=20)
    trainer.eval()

    model, bounds = codify_network(model, train_set.eat_other(test_set).to_dataframe(target=False))

    df = test_set.to_dataframe()
    len_inputs = []
    for index, instance in df.iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")
        # print(instance)
        inputs = get_miminal_explanation(model, instance, target, bounds, 3)
        len_inputs.append(len(inputs))
        print(inputs, "\n", len_inputs)
        break

    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")


def grid_search():
    devices = ['cpu']
    for device in devices:
        times = []
        for i in range(2):
            start = time.time()
            run()
            times.append(time.time() - start)

        print("Treinamento completo em:")
        print(f"Device: {device} | tempo: {sum(times)/2}")

if __name__ == '__main__':

    run()