import time

import numpy as np
import torch
from requests.packages import target
from sklearn.preprocessing import MinMaxScaler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Datasets.mnist.mnist_dataset_utils import get_dataframe_mnist
from src.back_explainer.milp.explainer import generate_explanation
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset
from src.legacy.codify_network import codify_network
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

def run(layers, relaxation=0.25):

    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])


    mnist_df = get_dataframe_mnist(target=False)

    mnist_network = ForwardReLU(layers)
    mnist_network.load_state_dict(torch.load(f'../../../Networks/mnist/Weights/mnist_net{layer_str}_weights01.pth',
                                             weights_only=True))


    start = time.perf_counter()
    mdl_relax, out_bounds_relax = relaxed_codify_network(mnist_network,
                                                         mnist_df,
                                                         relax_density=relaxation)
    print("Time to codify relaxed:", time.perf_counter() - start)

    print(out_bounds_relax)

    start = time.perf_counter()
    mdl , out_bounds = codify_network(mnist_network,
                                      mnist_df)
    print("Time to codify:", time.perf_counter() - start)
    print(out_bounds)


if __name__ == '__main__':
    list_layers = [[28*28, 16, 10],
                   [28*28, 16, 16, 10],
                   [28*28, 32, 10],
                   [28*28, 32, 32, 10],
                   [28*28, 64, 10],
                   [28*28, 64, 64, 10]]

    for layers in list_layers:
        print(f"Runnig codificator for: {layers}")
        run(layers)

