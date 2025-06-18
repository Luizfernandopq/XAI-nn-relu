from time import time

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

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

    wine_network = ForwardReLU([13, 13, 3])
    wine_network.load_state_dict(torch.load('wine_net_[13, 13, 3]_weights01.pth', weights_only=True))

    wine_network.eval()

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]

    all_set = train_set.eat_other(test_set)
    codificator = Codificator(wine_network, all_set.to_dataframe(target=False))
    bounds = codificator.codify_network_find_bounds()

    tamanhos = []
    for i in range(all_set.__len__()):
        print(f"Explain: {i}")
        instance, y_true = all_set[i]
        values = wine_network.get_all_neuron_values(instance)
        result = generate_explanation(
            np.argmax(values[-1]),
            weights,
            biases,
            values,
            bounds
        )
        print(f"Values: {values[0].tolist()}")
        print(f"Bounds: {bounds[0]}")
        print(f"Result: {result[0]}")
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
