from time import time, perf_counter

import numpy as np
import pandas as pd
import torch

from Datasets.sonar.sonar_dataset_utils import get_dataset_sonar
from src.legacy.explication import get_miminal_explanation

from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network, get_types_and_bounds


def test_fidelity(model, instance, inputs, prediction, domain):
    indexes = []
    explication = np.zeros(60, dtype=np.float32)

    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        indexes.append(index_input)
        explication[index_input] = 1.0

    # plot_explanation(instance.to_numpy(), explication)

    for i in range(len(instance)):
        if i not in indexes:
            lb, ub = domain[i]
            instance[i] = np.random.uniform(lb, ub)

    instance = torch.FloatTensor(instance)
    # print("OLD: ", prediction, end="\t|\t")
    # print("New: ", model(instance.unsqueeze(0)).argmax(dim=1).item(), end=" -> ")
    # plot_explanation(instance.numpy(), explication)

    if model(instance.unsqueeze(0)).argmax(dim=1).item() == prediction:
        return 1
    return 0


def run(layers, relax):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Data
    train_set, test_set = get_dataset_sonar()

    # Network and Train

    network = ForwardReLU(layers)
    network.load_state_dict(torch.load(f'../../Networks/sonar/Weights/sonar_net{layer_str}_weights.pth',
                                            weights_only=True))
    network.eval()
    all_set = train_set.eat_other(test_set)
    df = all_set.to_dataframe(target=False)

    start1 = time()
    relaxed_model, relaxed_bounds = relaxed_codify_network(network, df, relax_quatity=relax)

    print(f"Explicação iniciada após: {time() - start1}")
    _, domain = get_types_and_bounds(df)

    times = []
    sizes = []
    fidelities = 0



    for index, instance in df.iterrows():
        if index < 168:
            continue
        prediction = network(torch.FloatTensor(instance).unsqueeze(0)).argmax(dim=1).item()

        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, prediction, relaxed_bounds, 2)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        print(f"Explicado {index}: {perf_counter() - start}")

        fidelities += test_fidelity(network, instance, inputs, prediction, domain)

    fidelities = fidelities/len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")
    print(f"Fidelidade: {fidelities}")
    return times, sizes, fidelities


def append_results(experiments):
    df = pd.read_csv(f"../../Results/sonar.csv", index_col=0)
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)
    print(df)
    df.reset_index(drop=True).to_csv(f"../../Results/sonar.csv")

if __name__ == '__main__':

    list_layers = [#[60, 16, 16, 2],
                   #[60, 32, 32, 2],
                   #[60, 48, 48, 2],
                   #[60, 16, 16, 16, 2],
                   #[60, 32, 32, 32, 2],
                   #[60, 48, 48, 48, 2],
                   #[60, 16, 16, 16, 16, 2],
                   #[60, 32, 32, 32, 32, 2],
                   [60, 48, 48, 48, 48, 2]]

    relaxations = [0, 2, 4, 8]

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
            times, sizes, fidelitie = run(layers, relax)
            print(f"Tempo: {time() - start}")
            print()
            experiments["dataset"].append("sonar")
            experiments["network"].append(net_str)
            experiments["relaxation"].append(relax)
            experiments["time_mean"].append(np.mean(times))
            experiments["time_std"].append(np.std(times))
            experiments["expl_size_mean"].append(np.mean(sizes))
            experiments["expl_size_std"].append(np.std(sizes))
            experiments["fidelity"].append(fidelitie)

        append_results(experiments)
