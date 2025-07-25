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
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")
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
        [64, 48, 48, 48, 10],
        # [64, 16, 16, 16, 16, 10],
        [64, 32, 32, 32, 32, 10],
        [64, 48, 48, 48, 48, 10]]

    relaxations = [0, 2, 4, 8]

    samples = random.sample(range(0, 1797), 100)
    # relaxes = [7, 12, 45, 67, 75, 98, 131, 143, 167, 174, 187, 205, 213, 229, 231, 256, 257, 260, 284, 297, 306, 311, 346, 358, 362, 388, 419, 427, 431, 443, 445, 450, 451, 457, 458, 467, 468, 474, 485, 486, 500, 507, 511, 517, 519, 543, 564, 566, 567, 578, 589, 600, 613, 614, 620, 626, 644, 648, 662, 666, 682, 691, 692, 700, 719, 725, 731, 737, 748, 779, 797, 804, 806, 817, 829, 834, 837, 838, 844, 846, 849, 865, 867, 894, 922, 925, 936, 944, 950, 958, 959, 962, 964, 966, 972, 981, 1023, 1025, 1027, 1044, 1047, 1056, 1061, 1063, 1069, 1075, 1082, 1095, 1108, 1117, 1139, 1156, 1161, 1179, 1182, 1199, 1210, 1212, 1218, 1223, 1233, 1240, 1242, 1245, 1246, 1259, 1264, 1268, 1313, 1315, 1326, 1337, 1340, 1347, 1357, 1362, 1364, 1365, 1370, 1373, 1392, 1396, 1397, 1404, 1411, 1413, 1419, 1424, 1438, 1444, 1457, 1464, 1481, 1487, 1497, 1520, 1523, 1552, 1564, 1580, 1587, 1596, 1600, 1606, 1612, 1613, 1627, 1631, 1651, 1653, 1659, 1662, 1684, 1730, 1745, 1749, 1754, 1756, 1762, 1771]
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

        # append_results(experiments)
    # print(experiments)
    # experiments.to_csv(f"../../Results/digits.csv")