import torch

from Datasets.wine.wine_dataset_utils import get_dataset_wine
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.ForwardReluTrainer import ForwardReluTrainer

def run(layers):
    # Data
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    train_set, test_set = get_dataset_wine()

    # Network and Train

    wine_network = ForwardReLU(layers)
    trainer = ForwardReluTrainer(wine_network, train_loader=None)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(400)

    trainer.eval()
    torch.save(wine_network.state_dict(), f'../../../Networks/wine/Weights/wine_net{layer_str}_weights.pth')

if __name__ == '__main__':
    list_layers = [[13, 16, 16, 3],
                   [13, 32, 32, 3],
                   [13, 48, 48, 3],
                   [13, 16, 16, 16, 3],
                   [13, 32, 32, 32, 3],
                   [13, 48, 48, 48, 3],
                   [13, 16, 16, 16, 16, 3],
                   [13, 32, 32, 32, 32, 3],
                   [13, 48, 48, 48, 48, 3]]

    for layers in list_layers:
        run(layers)