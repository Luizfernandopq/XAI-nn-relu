import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from src.back_explainer.milp.explainer import generate_explanation
from src.back_explainer.milp.Codificator import Codificator
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.ForwardReluTrainer import ForwardReluTrainer
from src.back_explainer.network.SimpleDataset import SimpleDataset


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

    trainer.fit(epochs=10)
    trainer.eval()

    weights = [layer.weight.detach().numpy() for layer in model.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in model.layers if
              hasattr(layer, 'bias') and layer.bias is not None]

    codificator = Codificator(model, train_set.eat_other(test_set).to_dataframe(target=False))
    bounds = codificator.codify_network_find_bounds()
    print("Codificado")
    tamanhos = []
    for i in range(train_set.__len__()):
        print(f"Explain: {i}")
        instance, y_true = train_set[i]
        values = model.get_all_neuron_values(instance)
        result = generate_explanation(
            np.argmax(values[-1]),
            weights,
            biases,
            values,
            bounds
        )
        print(f"Values: {values[-3].tolist()}")
        print(f"Bounds: {bounds[-3]}")
        print(f"Result: {result[-2]}")
        tamanho = 0
        for i in range(len(instance)):
            if bounds[0][i][1] != result[0][i][1] or bounds[0][i][0] != result[0][i][0]:
                tamanho += 1
        tamanhos.append(tamanho)

    media = np.mean(tamanhos)
    mediana = np.median(tamanhos)
    maximo = np.max(tamanhos)
    minimo = np.min(tamanhos)

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