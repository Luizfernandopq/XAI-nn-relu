from time import time

import numpy as np
import torch

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.legacy.codify_network import codify_network
from src.legacy.explication import get_miminal_explanation
from src.modeler.network.ForwardReLU import ForwardReLU
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
    wine_network.load_state_dict(torch.load('wine_net_[13, 13, 3]_not_normal_weights01.pth', weights_only=True))

    wine_network.eval()

    model, bounds = codify_network(wine_network, train_set.eat_other(test_set).to_dataframe(target=False))

    df = test_set.to_dataframe()
    len_inputs = []
    start = time()
    for index, instance in df.iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")
        # print(instance)
        inputs = get_miminal_explanation(model, instance, target, bounds, 3)
        len_inputs.append(len(inputs))
        print(inputs, "\n", len_inputs)
    print(f"Tempo: {time() - start}")


    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")
