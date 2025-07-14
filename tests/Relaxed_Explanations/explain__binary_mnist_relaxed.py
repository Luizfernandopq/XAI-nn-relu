import random
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from Datasets.mnist.mnist_dataset_utils import get_dataframe_mnist_binary
from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.legacy.explication import get_miminal_explanation
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network, get_types_and_bounds


def plot_explanation(instance, explication):

    image = instance.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    # plt.close()

    image2 = explication.reshape(28, 28)
    plt.imshow(image2, cmap='RdGy')
    plt.title(f'Explicação:')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def steps_to_fidelity(model, instance, inputs, prediction, domain):
    important = []
    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        important.append(index_input)

    not_important = []

    for i in range(len(instance)):
        if i not in important:
            not_important.append(i)

    for count in range(len(not_important)):
        instance_copy = instance.copy()
        for i in not_important[:count+1]:
            lb, ub = domain[i]
            instance_copy[i] = np.random.uniform(lb, ub)
        instance_copy = torch.FloatTensor(instance_copy)

        if model(instance_copy.unsqueeze(0)).argmax(dim=1).item() != prediction:
            # plot_explanation(instance_copy.numpy(), instance.to_numpy())
            return count
    return len(not_important)


def test_fidelity(model, instance, inputs, prediction, domain):
    important = []
    explication = np.zeros(784, dtype=np.float32)
    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        important.append(index_input)
        explication[index_input] = 1.0

    not_important = []

    for i in range(len(instance)):
        if i not in important:
            not_important.append(i)

    # plot_explanation(instance.to_numpy(), explication)

    for i in not_important[:-115]:
        lb, ub = domain[i]
        instance[i] = np.random.uniform(lb, ub)

    instance = torch.FloatTensor(instance)
    # print("OLD: ", prediction, end="\t|\t")
    # print("New: ", model(instance.unsqueeze(0)).argmax(dim=1).item(), end=" -> ")
    # plot_explanation(instance.numpy(), explication)

    if model(instance.unsqueeze(0)).argmax(dim=1).item() == prediction:
        return 1
    return 0

def run(layers, relaxation, relaxes):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    mnist_df = get_dataframe_mnist_binary(target=False)

    mnist_network = ForwardReLU(layers)
    mnist_network.load_state_dict(torch.load(f'../../Networks/mnist_bin/Weights/mnist_net{layer_str}_weights.pth',
                                             weights_only=True))

    mnist_network.eval()

    start1 = time()
    relaxed_model, relaxed_bounds = relaxed_codify_network(mnist_network,
                                                           mnist_df,
                                                           relax_quatity=relaxation,
                                                           is_image=False)

    print(f"Explicação iniciada após: {time()-start1}")
    _, domain = get_types_and_bounds(mnist_df)
    times = []
    sizes = []
    fidelities = 0

    for index, instance in mnist_df.iterrows():
        if index not in relaxes:
            continue

        prediction = mnist_network(torch.FloatTensor(instance).unsqueeze(0)).argmax(dim=1).item()
        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, prediction, relaxed_bounds, 2)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        fid = test_fidelity(mnist_network, instance, inputs, prediction, domain)
        fidelities += fid
        true_fidelity = steps_to_fidelity(mnist_network, instance, inputs, prediction, domain)
        print(f"Checkpoint Explicado {len(times)}: {perf_counter() - start} | Tamanho: {len(inputs)}"
              f" | Fidelidade Relativa {784 - true_fidelity}"
              f" | Devolvidas: {784 - (len(inputs) + true_fidelity)}")
        # fidelities += 784 - true_fidelity

    fidelities = fidelities / len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo} | Fidelidade: {fidelities}")
    return times, sizes, fidelities

def append_results(experiments):
    df = pd.read_csv(f"../../Results/mnist_bin.csv", index_col=0)
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)
    print(df)
    df.to_csv(f"../../Results/mnist_bin.csv")

if __name__ == '__main__':

    list_layers = [[28 * 28, 16, 16, 2],
                   [28 * 28, 32, 32, 2],
                   [28 * 28, 16, 16, 16, 2],
                   [28 * 28, 16, 16, 16, 16, 2]]


    relaxations = [0, 2, 4, 8]


    relaxes = random.sample(range(0, 2000), 100)
    # relaxes = [1333,1758, 2873, 2953, 3006, 3076, 3287, 3738, 3784, 4694, 4719]
    # relaxes.append(1333)
    # relaxes.append(402)
    print(sorted(relaxes))
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
            times, sizes, fidelitie = run(layers, relax, relaxes)
            print(f"Tempo: {time() - start}")
            print()

            experiments["dataset"].append("mnsit_bin")
            experiments["network"].append(net_str)
            experiments["relaxation"].append(relax)
            experiments["time_mean"].append(np.mean(times))
            experiments["time_std"].append(np.std(times))
            experiments["expl_size_mean"].append(np.mean(sizes))
            experiments["expl_size_std"].append(np.std(sizes))
            experiments["fidelity"].append(fidelitie)

        append_results(experiments)
    # experiments = pd.DataFrame(experiments)
    # print(experiments)
    # experiments.to_csv(f"../../Results/mnist.csv")