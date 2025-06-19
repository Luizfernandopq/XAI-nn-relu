import pickle
from time import time

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.legacy.codify_network import codify_network
from src.legacy.explication import get_miminal_explanation

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network


def run(layers=None):
    if layers is None:
        layers = [13, 16, 3]

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

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

    wine_network = ForwardReLU(layers)
    wine_network.load_state_dict(torch.load(f'../../Networks/wine/Weights/wine_net{layer_str}_weights01.pth',
                                            weights_only=True))


    wine_network.eval()

    all_set = train_set.eat_other(test_set)
    df = all_set.to_dataframe()

    relaxed_model, relaxed_bounds = relaxed_codify_network(wine_network, all_set.to_dataframe(target=False),
                                                           relax_density=0.5)

    len_inputs = []
    start = time()
    for index, instance in df.iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")
        # print(instance)
        inputs = get_miminal_explanation(relaxed_model, instance, target, relaxed_bounds, 3)
        len_inputs.append(len(inputs))
        # print(inputs, "\n", len_inputs)

    print(f"Tempo Relaxed: {time() - start}")
    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")

    # ------------------------------------ PART 2 ----------------------------------------

    model, bounds_legacy = codify_network(wine_network, all_set.to_dataframe(target=False))

    len_inputs = []

    start = time()
    for index, instance in df.iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")
        # print(instance)
        inputs = get_miminal_explanation(model, instance, target, bounds_legacy, 3)
        len_inputs.append(len(inputs))
        # print(inputs, "\n", len_inputs)
    print(f"Tempo legacy: {time() - start}")

    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")


if __name__ == '__main__':
    list_layers = [[13, 16, 3],
                   [13, 16, 16, 3],
                   [13, 32, 3],
                   [13, 32, 32, 3]]
    for layers in list_layers:
        print(f"Rodando: {layers}")
        run(layers)