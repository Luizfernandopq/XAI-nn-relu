import itertools

import torch

from Datasets.digits.digits_dataset_utils import get_dataset_digits
from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.network.ForwardReluTrainer import ForwardReluTrainer

def run(layers, hparams):
    # Data
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = get_dataset_digits()

    # Network and Train
    digits_network = ForwardReLU(layers)
    optimizer = torch.optim.Adam(
        digits_network.parameters(),
        lr=hparams["lr"],
        betas=hparams["betas"],
        weight_decay=hparams["weight_decay"],
    )
    trainer = ForwardReluTrainer(digits_network,
                                 train_loader=None,
                                 optimizer=optimizer,
                                 device=device)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(60)

    acc = trainer.eval()
    if acc > 0.97:
        torch.save(digits_network.state_dict(), f'../../../Networks/digits/Weights/digits_net{layer_str}_weights.pth')
    return acc

if __name__ == '__main__':
    list_layers = [#[64, 16, 16, 10],
                   #[64, 32, 32, 10],
                   #[64, 48, 48, 10],
                   #[64, 16, 16, 16, 10],
                   #[64, 32, 32, 32, 10],
                   #[64, 48, 48, 48, 10],
                   #[64, 16, 16, 16, 16, 10],
                   [64, 32, 32, 32, 32, 10],
                   [64, 48, 48, 48, 48, 10]]
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
            acc = run(layers=layer, hparams=hparams)
            print(f"{layer} Acurácia: {acc:.4f} - Hparams: {hparams}\n")
            if acc > best_acc:
                best_acc = acc
                best_params = hparams
            if acc > 0.99:
                break

        print("\nMelhor acurácia:", best_acc)
        print(layer)
        print("Melhores parâmetros:", best_params)
