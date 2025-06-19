import torch

from Datasets.wine.wine_dataset_utils import get_dataset_wine
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.ForwardReluTrainer import ForwardReluTrainer


if __name__ == '__main__':

    # Data

    train_set, test_set = get_dataset_wine()

    # Network and Train

    layers = [13, 16, 16, 3]
    wine_network = ForwardReLU(layers)
    trainer = ForwardReluTrainer(wine_network, train_loader=None)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(400)

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]
    print(weights)
    i = int(input("Network number: "))
    torch.save(wine_network.state_dict(), f'../../../Networks/N-Weights/wine_net_13x16x16x3_weights{i:02}.pth')
