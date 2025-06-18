from time import time

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.back_explainer.milp.explainer import generate_explanation
from src.back_explainer.milp.Codificator import Codificator

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset

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

    wine_network = ForwardReLU([13, 8, 8, 8, 8, 3])
    wine_network.load_state_dict(torch.load('wine_net_[13, 8, 8, 8, 8, 3]_weights01.pth', weights_only=True))

    wine_network.eval()

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]

    codificator = Codificator(wine_network, train_set.eat_other(test_set).to_dataframe(target=False))
    bounds = codificator.codify_network_find_bounds()

    start = time()
    for i in range(train_set.__len__()):
        instance, y_true = train_set[i]
        values = wine_network.get_all_neuron_values(instance)
        result = generate_explanation(
            np.argmax(values[-1]),
            weights,
            biases,
            values,
            bounds
        )

    print(f"Tempo: {time() - start}")

