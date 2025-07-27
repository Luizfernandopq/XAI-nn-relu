import random
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from Datasets.digits.digits_dataset_utils import get_dataset_digits
from src.legacy.explication import get_miminal_explanation

from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network, get_types_and_bounds


def plot_explanation(instance, explication):

    image = instance.reshape(8, 8)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(4)
    # plt.close()

    image2 = explication.reshape(8, 8)
    plt.imshow(image2, cmap='RdGy')
    plt.title(f'Explicação:')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(4)
    plt.close()

# def test_fidelity(model, instance, inputs, prediction):
#     indexes = []
#     for j in inputs:
#         index_input = int(j.name.split("input")[1]) - 1
#         indexes.append(index_input)
#     for i in range(len(instance)):
#         if i not in indexes:
#             instance[i] = np.random.uniform(0.0, 1.0)
#     if model(torch.FloatTensor(instance).unsqueeze(0)).argmax(dim=1).item() == prediction:
#         return 1
#     return 0

def test_fidelity(model, instance, inputs, prediction, domain):
    indexes = []
    explication = np.zeros(64, dtype=np.float32)

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

def run(layers, relax, samples):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Data
    train_set, test_set = get_dataset_digits()

    # Network and Train

    network = ForwardReLU(layers)
    network.load_state_dict(torch.load(f'../../Networks/digits/Weights/digits_net{layer_str}_weights.pth',
                                            weights_only=True))
    network.eval()
    all_set = train_set.eat_other(test_set)
    df = all_set.to_dataframe(target=False)

    start1 = time()
    relaxed_model, relaxed_bounds = relaxed_codify_network(network, df, relax_quatity=relax)

    relaxed_model.parameters.timelimit = 300

    print(f"Explicação iniciada após: {time() - start1}")
    _, domain = get_types_and_bounds(df)

    times = []
    sizes = []
    fidelities = 0


    for index, instance in df.iterrows():
        if index not in samples:
            continue

        prediction = network(torch.FloatTensor(instance).unsqueeze(0)).argmax(dim=1).item()

        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, prediction, relaxed_bounds, 10)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        # print(f"Explicado {index}: {perf_counter() - start}")

        fid = test_fidelity(network, instance, inputs, prediction, domain)
        fidelities += fid
        if len(times) % 5 == 0:
            print(f"Checkpoint Explicado {len(times)}: {perf_counter() - start} | média: {np.mean(times)} | "
                  f"Fidelidade {fidelities / len(times)}")
    fidelities = fidelities/len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"EXPLICAÇÃO -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")
    media = np.mean(times)
    mediana = np.median(times)
    maximo = np.max(times)
    minimo = np.min(times)
    print(f"TEMPO -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")

    print(f"Fidelidade: {fidelities}")
    return times, sizes, fidelities

def append_results(experiments):
    df = pd.read_csv(f"../../Results/digits.csv", index_col=0)
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)
    print(df)
    df.to_csv(f"../../Results/digits.csv")

if __name__ == '__main__':

    list_layers = [#[64, 16, 16, 10],
        # [64, 32, 32, 10],
        # [64, 48, 48, 10],
        # [64, 16, 16, 16, 10],
        # [64, 32, 32, 32, 10],
        #[64, 48, 48, 48, 10],
        # [64, 16, 16, 16, 16, 10],
        [64, 32, 32, 32, 32, 10],
        ]#[64, 48, 48, 48, 48, 10]]

    relaxations = [0, 2, 4, 8]

    # samples = random.sample(range(0, 1797), 100)
    samples = [10, 14, 25, 47, 51, 63, 68, 102, 108, 109, 116, 126, 149, 173, 227, 228, 255, 260, 264, 302, 320, 340, 346, 355, 389, 394, 453, 455, 469, 471, 494, 506, 540, 547, 548, 559, 568, 578, 588, 592, 601, 611, 617, 627, 637, 644, 656, 673, 679, 740, 751, 800, 831, 854, 857, 885, 892, 914, 940, 945, 970, 1017, 1048, 1086, 1101, 1152, 1180, 1188, 1246, 1247, 1294, 1316, 1338, 1343, 1381, 1385, 1418, 1439, 1441, 1447, 1521, 1532, 1547, 1550, 1576, 1579, 1591, 1598, 1660, 1661, 1674, 1685, 1690, 1737, 1739, 1762, 1765, 1790, 1792, 1794]

    # relaxes = relaxes[70:72]
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
            print()
            experiments["dataset"].append("digits")
            experiments["network"].append(net_str)
            experiments["relaxation"].append(relax)
            experiments["time_mean"].append(np.mean(times))
            experiments["time_std"].append(np.std(times))
            experiments["expl_size_mean"].append(np.mean(sizes))
            experiments["expl_size_std"].append(np.std(sizes))
            experiments["fidelity"].append(fidelitie)

        append_results(experiments)
    # print(experiments)
    # experiments.to_csv(f"../../Results/digits.csv")