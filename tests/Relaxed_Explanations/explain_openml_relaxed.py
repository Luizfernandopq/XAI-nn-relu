import random
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch

from Datasets.openml_datasets import get_dataset_openml
from src.legacy.explication import get_miminal_explanation

from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network, get_types_and_bounds


def test_fidelity(model, instance, inputs, prediction, domain):
    indexes = []

    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        indexes.append(index_input)

    for i in range(len(instance)):
        if i not in indexes:
            lb, ub = domain[i]
            instance[i] = np.random.uniform(lb, ub)

    instance = torch.FloatTensor(instance)
    if model(instance.unsqueeze(0)).argmax(dim=1).item() == prediction:
        return 1
    return 0


def run(layers, relax, dataset_name, df, samples):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Data

    # Network and Train

    network = ForwardReLU(layers)
    network.load_state_dict(torch.load(f'../../Networks/{dataset_name}/{dataset_name}_net{layer_str}_weights.pth',
                                            weights_only=True))
    network.eval()

    start1 = time()
    relaxed_model, relaxed_bounds = relaxed_codify_network(network, df, relax_quatity=relax)
    relaxed_model.parameters.timelimit = 300

    print(f"Explicação iniciada após: {time() - start1}")
    _, domain = get_types_and_bounds(df)

    times = []
    sizes = []
    fidelities = 0

    relaxed_model.parameters.timelimit = 300

    for index, instance in df.iterrows():
        if index not in samples:
            continue
        prediction = network(torch.FloatTensor(instance).unsqueeze(0)).argmax(dim=1).item()

        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, prediction, relaxed_bounds, layers[-1])
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        if len(times) % 5 == 0:
            print(f"Explicado {len(times)}: {perf_counter() - start}")

        fidelities += test_fidelity(network, instance, inputs, prediction, domain)

    fidelities = fidelities/len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")

    print(f"TEMPO -> Média: {np.mean(times)}, Mediana: {np.median(times)},"
          f" Máximo: {np.max(times)}, Mínimo: {np.min(times)}")

    print(f"Fidelidade: {fidelities}")
    return times, sizes, fidelities


def append_results(experiments, dataset_name):
    try:
        df = pd.read_csv(f"../../Results/{dataset_name}.csv", index_col=0)
    except:
        df = pd.DataFrame()
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)
    print(df)
    df.reset_index(drop=True).to_csv(f"../../Results/{dataset_name}.csv")

def explain(dataset_name):
    list_layers = [[60, 16, 16, 2],
                   [60, 32, 32, 2],
                   # [60, 48, 48, 2],
                   [60, 16, 16, 16, 2],
                   [60, 32, 32, 32, 2],
                   # [60, 48, 48, 48, 2],
                   [60, 16, 16, 16, 16, 2],
                   [60, 32, 32, 32, 32, 2],
                   ]#[60, 48, 48, 48, 48, 2]]

    relaxations = [0, 2, 4, 8]

    train_set, test_set = get_dataset_openml(dataset_name)
    df = train_set.eat_other(test_set).to_dataframe(target=False)

    heart = [4, 5, 6, 9, 11, 14, 16, 18, 19, 20, 21, 22, 23, 24, 31, 32, 35, 36, 37, 38, 50, 51, 52, 55, 56, 57, 58, 60, 61, 62, 63, 66, 67, 70, 71, 73, 74, 78, 80, 84, 90, 95, 96, 98, 100, 106, 113, 116, 118, 120, 121, 122, 128, 129, 130, 135, 136, 139, 140, 144, 148, 149, 151, 155, 156, 157, 170, 171, 174, 177, 184, 185, 186, 190, 199, 200, 201, 207, 208, 212, 216, 217, 218, 221, 222, 224, 232, 234, 236, 237, 239, 240, 245, 248, 251, 256, 259, 266, 267, 268]

    samples = random.sample(range(0, len(df)), 100)
    print(len(samples), sorted(samples))
    for layers in list_layers:
        layers[0] = train_set.X.shape[1]
        layers[-1] = torch.max(train_set.y).item() + 1
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

        for relax in relaxations:
            net_str = f"Net_{len(layers) - 2}x{layers[1]}_hidden"

            print(f"Rodando: {layers} relax: {relax}")
            start = time()
            times, sizes, fidelitie = run(layers, relax, dataset_name, df, samples)
            print(f"Tempo: {time() - start}")
            print()
            experiments["dataset"].append(f"{dataset_name}")
            experiments["network"].append(net_str)
            experiments["relaxation"].append(relax)
            experiments["time_mean"].append(np.mean(times))
            experiments["time_std"].append(np.std(times))
            experiments["expl_size_mean"].append(np.mean(sizes))
            experiments["expl_size_std"].append(np.std(sizes))
            experiments["fidelity"].append(fidelitie)

        append_results(experiments, dataset_name)


if __name__ == '__main__':
    nets = ["diabetes", "glass", "heart-statlog", "iris"]
    nets.pop(0)
    nets.pop(0)
    nets.pop(0)

    for net in nets:
        print(net)
        explain(net)