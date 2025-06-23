import random
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from Datasets.mnist.mnist_dataset_utils import get_dataframe_mnist
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.legacy.explication import get_miminal_explanation
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network, get_types_and_bounds


def plot_explanation(instance, explication):

    image = instance.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(6)
    # plt.close()

    image2 = explication.reshape(28, 28)
    plt.imshow(image2, cmap='RdGy')
    plt.title(f'Explicação:')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(6)
    plt.close()

def test_fidelity(model, instance, inputs, predicition, domain):
    indexes = []
    explication = np.zeros(784, dtype=np.float32)
    for j in inputs:
        index_input = int(j.name.split("input")[1]) - 1
        indexes.append(index_input)
        explication[index_input] = 1.0

    # plot_explanation(instance.to_numpy(), explication)

    for i in range(len(instance)):
        if i not in indexes:
            lb, ub = domain[i]
            instance[i] = np.random.uniform(lb, ub)

    # plot_explanation(instance.numpy(), explication)
    instance = torch.FloatTensor(instance)

    if model(instance.unsqueeze(0)).argmax(dim=1).item() == predicition:
        return 1
    return 0

def run(layers, relaxation, relaxes):

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    mnist_df = get_dataframe_mnist(target=False)

    mnist_network = ForwardReLU(layers)
    mnist_network.load_state_dict(torch.load(f'../../Networks/mnist/Weights/mnist_net{layer_str}_weights.pth',
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
        inputs = get_miminal_explanation(relaxed_model, instance, prediction, relaxed_bounds, 10)
        times.append(perf_counter() - start)
        sizes.append(len(inputs))
        if len(times) % 20 == 1:
            print(f"Checkpoint Explicado {len(times)}: {perf_counter() - start} | média: {np.mean(times)}")
        fidelities += test_fidelity(mnist_network, instance, inputs, prediction, domain)

    fidelities = fidelities / len(sizes)

    media = np.mean(sizes)
    mediana = np.median(sizes)
    maximo = np.max(sizes)
    minimo = np.min(sizes)
    print(f"Tamanho -> Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo} | Fidelidade: {fidelities}")
    return times, sizes, fidelities

def append_results(experiments):
    df = pd.read_csv(f"../../Results/mnist.csv", index_col=0)
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)
    print(df)
    df.to_csv(f"../../Results/mnist.csv")

if __name__ == '__main__':

    list_layers = [#[28 * 28, 16, 16, 10],
                   #[28 * 28, 32, 32, 10],
                   [28 * 28, 16, 16, 16, 10],
                   [28 * 28, 16, 16, 16, 16, 10]]


    relaxations = [0, 2, 4, 8]

    # relaxes = random.sample(range(0, 10000), 100)
    relaxes = [62, 99, 309, 378, 641, 794, 877, 1006, 1041, 1180, 1229, 1361, 1583, 1672, 1795, 1914, 2011, 2181, 2395, 2414, 2427, 2592, 2686, 2745, 2766, 2821, 3021, 3193, 3251, 3284, 3389, 3461, 3512, 3731, 3746, 4030, 4171, 4224, 4356, 4364, 4451, 4540, 4729, 4800, 4857, 4884, 4929, 5006, 5084, 5126, 5201, 5500, 5540, 5743, 5876, 5979, 6104, 6212, 6358, 6628, 6947, 6955, 7092, 7173, 7214, 7380, 7406, 7447, 7591, 7815, 7865, 7965, 8107, 8246, 8440, 8504, 8519, 8749, 8757, 8767, 8809, 8828, 8890, 8958, 9045, 9081, 9182, 9331, 9336, 9384, 9429, 9441, 9484, 9583, 9711, 9725, 9787, 9802, 9831, 9990]

    # relaxes = [402]
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

            experiments["dataset"].append("mnsit")
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