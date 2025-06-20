from time import time, perf_counter

import numpy as np
import pandas as pd
import torch


from Datasets.wine.wine_dataset_utils import get_dataset_wine
from src.legacy.explication import get_miminal_explanation

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

def test_fidelity(model, instance, inputs, target):
    indexes = []
    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        indexes.append(index_input)
    for i in range(len(instance)):
        if i not in indexes:
            instance[i] = np.random.rand()
    if model(instance.unsqueeze(0)).argmax(dim=1).item() == target:
        return 1
    return 0


def run(layers, relax):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Data
    train_set, test_set = get_dataset_wine()

    # Network and Train

    wine_network = ForwardReLU(layers)
    wine_network.load_state_dict(torch.load(f'../../Networks/wine/Weights/wine_net{layer_str}_weights.pth',
                                            weights_only=True))
    wine_network.eval()
    all_set = train_set.eat_other(test_set)
    df = all_set.to_dataframe()

    times = []
    sizes = []
    fidelities = 0


    relaxed_model, relaxed_bounds = relaxed_codify_network(wine_network, all_set.to_dataframe(target=False),
                                                           relax_density=relax)

    for index, instance in df.iterrows():
        target = instance["target"].astype(int)
        instance = instance.drop("target")

        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, target, relaxed_bounds, 3)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))

        fidelities += test_fidelity(wine_network, all_set[index][0], inputs, target)

    fidelities = fidelities/len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")
    print(f"Fidelidade: {fidelities}")
    return times, sizes, fidelities


if __name__ == '__main__':
    experiments = {
        "dataset": [],
        "network": [],
        "relaxation": [],
        "time_mean": [],
        "time_std": [],
        "expl_size_mean": [],
        "expl_size_std": [],
        "fidelity": [],
    }

    list_layers = [[13, 16, 16, 3],
                   [13, 32, 32, 3],
                   [13, 48, 48, 3],
                   [13, 16, 16, 16, 3],
                   [13, 32, 32, 32, 3],
                   [13, 48, 48, 48, 3],
                   [13, 16, 16, 16, 16, 3],
                   [13, 32, 32, 32, 32, 3],
                   [13, 48, 48, 48, 48, 3]]

    relaxations = [0.0, 0.25, 0.45, 0.60]

    for layers in list_layers:
        for relax in relaxations:
            net_str = f"Net_{len(layers) - 2}x{layers[1]}_hidden"

            print(f"Rodando: {net_str} relax: {relax}")
            start = time()
            times, sizes, fidelitie = run(layers, relax)
            print(f"Tempo: {time() - start}")
            experiments["dataset"].append("wine")
            experiments["network"].append(net_str)
            experiments["relaxation"].append(relax)
            experiments["time_mean"].append(np.mean(times))
            experiments["time_std"].append(np.std(times))
            experiments["expl_size_mean"].append(np.mean(sizes))
            experiments["expl_size_std"].append(np.std(sizes))
            experiments["fidelity"].append(fidelitie)

    experiments = pd.DataFrame(experiments)
    print(experiments)
    experiments.to_csv(f"../../Results/wine.csv")