import pickle
from time import time

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.legacy.codify_network import codify_network
from src.legacy.explication import get_miminal_explanation
from src.back_explainer.milp.explainer import generate_explanation

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset

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
    wine_network.load_state_dict(torch.load(f'../../../Networks/wine/Weights/wine_net{layer_str}_weights01.pth',
                                            weights_only=True))


    wine_network.eval()

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]

    with open(f"../../../Networks/wine/Bounds/wine_bound{layer_str}.pkl", "rb") as f:
        bounds = pickle.load(f)

    len_inputs = []

    all_set = train_set.eat_other(test_set)
    start = time()
    for i in range(all_set.__len__()):
        instance, y_true = all_set[i]
        values = wine_network.get_all_neuron_values(instance)
        result, inputs = generate_explanation(
            np.argmax(values[-1]),
            weights,
            biases,
            values,
            bounds
        )
        len_inputs.append(len(inputs))

    print(f"Tempo new: {time() - start}")
    media = np.mean(len_inputs)
    mediana = np.median(len_inputs)
    maximo = np.max(len_inputs)
    minimo = np.min(len_inputs)

    print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")

    # model = ModelReader.read(f"../../../Networks/wine/Milps/wine_milp{layer_str}.lp")

    model, bounds_legacy = codify_network(wine_network, all_set.to_dataframe(target=False))

    df = all_set.to_dataframe()

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

    print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")


if __name__ == '__main__':
    list_layers = [[13, 16, 3],
                   [13, 16, 16, 3],
                   [13, 32, 3],
                   [13, 32, 32, 3]]
    for layers in list_layers:
        print(f"Rodando: {layers}")
        run(layers)