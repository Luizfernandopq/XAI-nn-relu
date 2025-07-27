import random
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch

from Datasets.breast_cancer.breast_cancer_dataset_utils import get_dataset_breast_cancer
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


def run(layers, relax, samples):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Data
    train_set, test_set = get_dataset_breast_cancer()

    # Network and Train

    breast_cancer_network = ForwardReLU(layers)
    breast_cancer_network.load_state_dict(torch.load(f'../../Networks/breast_cancer/Weights/breast_cancer_net{layer_str}_weights.pth',
                                            weights_only=True))
    breast_cancer_network.eval()
    all_set = train_set.eat_other(test_set)
    df = all_set.to_dataframe(target=False)
    start1 = time()
    relaxed_model, relaxed_bounds = relaxed_codify_network(breast_cancer_network, df, relax_quatity=relax)
    _, domain = get_types_and_bounds(df)

    relaxed_model.parameters.timelimit = 300

    print(f"Explicação iniciada após: {time() - start1}")

    times = []
    sizes = []
    fidelities = 0

    for index, instance in df.iterrows():
        if index not in samples:
            continue

        prediction = breast_cancer_network(torch.FloatTensor(instance).unsqueeze(0)).argmax(dim=1).item()

        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, prediction, relaxed_bounds, 2)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        if len(times) % 20 == 0:
            print(f"Checkpoint Explicado {len(times)}: {perf_counter() - start} | média: {np.mean(times)} | "
                  f"Fidelidade {fidelities / len(times)}")
        fidelities += test_fidelity(breast_cancer_network, instance, inputs, prediction, domain)

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

def append_results(experiments):
    df = pd.read_csv(f"../../Results/breast_cancer.csv", index_col=0)
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)
    print(df)
    df.reset_index(drop=True).to_csv(f"../../Results/breast_cancer.csv")


if __name__ == '__main__':

    list_layers = [#[30, 16, 16, 2],
                   # [30, 32, 32, 2],
                   # [30, 48, 48, 2],
                   # [30, 16, 16, 16, 2],
                   # [30, 32, 32, 32, 2],
                   # [30, 48, 48, 48, 2],
                   # [30, 16, 16, 16, 16, 2],
                   [30, 32, 32, 32, 32, 2],
                   ]#[30, 48, 48, 48, 48, 2]]

    samples = random.sample(range(0, 569), 100)

    relaxations = [0, 2, 4, 8]
    print(sorted(samples))
    for layers in list_layers:
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

            print(f"Rodando: {net_str} relax: {relax}")
            start = time()
            times, sizes, fidelitie = run(layers, relax, samples)
            print(f"Tempo: {time() - start}")
            experiments["dataset"].append("breast_cancer")
            experiments["network"].append(net_str)
            experiments["relaxation"].append(relax)
            experiments["time_mean"].append(np.mean(times))
            experiments["time_std"].append(np.std(times))
            experiments["expl_size_mean"].append(np.mean(sizes))
            experiments["expl_size_std"].append(np.std(sizes))
            experiments["fidelity"].append(fidelitie)

        append_results(experiments)