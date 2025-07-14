import itertools

import torch

from Datasets.sonar.sonar_dataset_utils import get_dataset_sonar
from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.network.ForwardReluTrainer import ForwardReluTrainer

def run(layers, hparam):
    # Data
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    train_set, test_set = get_dataset_sonar()

    # Network and Train

    sonar_network = ForwardReLU(layers)
    optimizer = torch.optim.Adam(
        sonar_network.parameters(),
        lr=hparam["lr"],
        betas=hparam["betas"],
        weight_decay=hparam["weight_decay"],
    )
    trainer = ForwardReluTrainer(sonar_network, train_loader=None, optimizer=optimizer)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(400)

    acc = trainer.eval()
    print("ACCC", acc)
    if acc > 0.91:
        torch.save(sonar_network.state_dict(), f'../../../Networks/sonar/Weights/sonar_net{layer_str}_weights.pth')
    return acc

if __name__ == '__main__':
    list_layers = [[60, 16, 16, 2],
                   [60, 32, 32, 2],
                   [60, 48, 48, 2],
                   [60, 16, 16, 16, 2],
                   [60, 32, 32, 32, 2],
                   [60, 48, 48, 48, 2],
                   [60, 16, 16, 16, 16, 2],
                   [60, 32, 32, 32, 32, 2],
                   [60, 48, 48, 48, 48, 2]]

    param_grid = {
        "lr": [1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [0.0, 1e-4, 1e-5],
        "betas": [(0.9, 0.999), (0.85, 0.995), (0.85, 0.999), (0.8, 0.99)]
    }

    all_params = list(itertools.product(*param_grid.values()))
    best_acc = 0
    best_params = None
    for layer in list_layers:
        best_acc = 0
        best_params = None
        for param_set in all_params:
            hparams = dict(zip(param_grid.keys(), param_set))
            acc = run(layers=layer, hparam=hparams)
            # print(f"Acurácia: {acc:.4f} - Hparams: {hparams}")
            if acc > best_acc:
                best_acc = acc
                best_params = hparams

        print("\nMelhor acurácia:", best_acc)
        print(layer)
        print("Melhores parâmetros:", best_params)