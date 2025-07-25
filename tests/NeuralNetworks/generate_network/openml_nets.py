import itertools
import time

import numpy as np
import torch

from Datasets.openml_datasets import get_dataset_openml
from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.network.ForwardReluTrainer import ForwardReluTrainer

def run(layers, train_set, test_set, dataset_name, hparams, epochs=820, best=0):
    # Data
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Network and Train
    network = ForwardReLU(layers)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=hparams["lr"],
        betas=hparams["betas"],
        weight_decay=hparams["weight_decay"],
    )
    trainer = ForwardReluTrainer(network,
                                 train_loader=None,
                                 optimizer=optimizer,
                                 device=device)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(epochs)

    # print(f"NET: {layer_str}: {epochs} epochs", end='\t')
    acc = trainer.eval(verbose=0)
    if best < acc:
        best = acc
    if acc > 0.79:
        torch.save(network.state_dict(), f'../../../Networks/{dataset_name}/{dataset_name}_net{layer_str}_weights.pth')
        return acc
    if epochs == 20:
        return best
    else:
        return run(layers, train_set, test_set, dataset_name, hparams, epochs-200, best)

def train_network(dataset_name):
    print(dataset_name)
    param_grid = {
        "lr": [1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [0.0, 1e-4, 1e-5],
        "betas": [(0.9, 0.999), (0.85, 0.995), (0.85, 0.999), (0.8, 0.99)]
    }
    list_layers = [#[10, 16, 16, 2],
                   # [10, 32, 32, 2],
                   # [10, 48, 48, 2],
                   [10, 16, 16, 16, 2],
                   # [10, 32, 32, 32, 2],
                   # [10, 48, 48, 48, 2],
                   # [10, 16, 16, 16, 16, 2],
                   # [10, 32, 32, 32, 32, 2],
                   ]#[10, 48, 48, 48, 48, 2]]

    train, test = get_dataset_openml(dataset_name)

    all_params = list(itertools.product(*param_grid.values()))
    for layer in list_layers:
        layer[0] = train.X.shape[1]
        layer[-1] = torch.max(train.y).item() + 1

        best_acc = 0
        best_params = None
        start = time.time()
        for param_set in all_params:
            hparams = dict(zip(param_grid.keys(), param_set))
            acc = run(layer, train, test, dataset_name, hparams)

            # print(f"{layer} Acurácia: {acc:.4f} - Hparams: {hparams}")
            # print(time.time() - start, "\n")
            if acc > best_acc:
                best_acc = acc
                best_params = hparams
            if acc > 0.82:
                break

        print("\nMelhor acurácia:", best_acc)
        print(layer)
        print("Melhores parâmetros:", best_params)


if __name__ == '__main__':
    nets = ["diabetes", "glass", "heart-statlog"]
    nets.pop(2)
    nets.pop(0)
    for net in nets:
        train_network(net)