import random
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from Datasets.mnist.mnist_dataset_utils import get_dataframe_mnist
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.legacy.explication import get_miminal_explanation
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

def plot_explanation(instance, explication):

    image = instance.to_numpy().reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(4)
    # plt.close()

    image2 = explication.reshape(28, 28)
    plt.imshow(image2, cmap='RdGy')
    plt.title(f'Explicação:')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(4)
    plt.close()

def test_fidelity(model, instance, inputs, target):
    indexes = []
    # explication = np.zeros(784, dtype=np.float32)
    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        indexes.append(index_input)
    #     explication[index_input] = 1.0
    # plot_explanation(instance, explication)

    instance = torch.FloatTensor(instance)
    for i in range(len(instance)):
        if i not in indexes:
            instance[i] = np.random.rand()
    if model(instance.unsqueeze(0)).argmax(dim=1).item() == target:
        return 1
    return 0

def run(layers, relaxation, relaxes):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    mnist_df = get_dataframe_mnist(target=True)

    mnist_network = ForwardReLU(layers)
    mnist_network.load_state_dict(torch.load(f'../../Networks/mnist/Weights/mnist_net{layer_str}_weights.pth',
                                             weights_only=True))

    mnist_network.eval()

    start1 = time()
    relaxed_model, relaxed_bounds = relaxed_codify_network(mnist_network, mnist_df.drop("target", axis=1),
                                                           relax_density=relaxation)

    print(f"Explicação iniciada após: {time()-start1}")
    times = []
    sizes = []
    fidelities = 0

    for index, instance in mnist_df.iterrows():
        if index not in relaxes:
            continue
        target = instance["target"].astype(int)
        instance = instance.drop("target")

        start = perf_counter()
        inputs = get_miminal_explanation(relaxed_model, instance, target, relaxed_bounds, 10)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        print(f"Explicado {index}: {perf_counter() - start}")
        fidelities += test_fidelity(mnist_network, instance, inputs, target)


    fidelities = fidelities / len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo} | Fidelidade: {fidelities}")
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
    list_layers = [[28 * 28, 16, 16, 10],
                   [28 * 28, 32, 32, 10],
                   [28 * 28, 16, 16, 16, 10],
                   [28 * 28, 16, 16, 16, 16, 10]]

    relaxations = [0.0, 0.075, 0.15, 0.20]

    relaxes = random.sample(range(0, 10000), 10)
    print(sorted(relaxes))
    for layers in list_layers:

        for relax in relaxations:
            net_str = f"Net_{len(layers) - 2}x{layers[1]}_hidden"
            try:
                print(f"Rodando: {net_str} relax: {relax}")
                start = time()
                times, sizes, fidelitie = run(layers, relax, relaxes)
                print(f"Tempo: {time() - start}")
                print()

                experiments["dataset"].append("wine")
                experiments["network"].append(net_str)
                experiments["relaxation"].append(relax)
                experiments["time_mean"].append(np.mean(times))
                experiments["time_std"].append(np.std(times))
                experiments["expl_size_mean"].append(np.mean(sizes))
                experiments["expl_size_std"].append(np.std(sizes))
                experiments["fidelity"].append(fidelitie)
            except:
                print(f"Falhou com: {layers}")

    experiments = pd.DataFrame(experiments)
    print(experiments)
    experiments.to_csv(f"../../Results/mnist.csv")