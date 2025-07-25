import torch

from Datasets.breast_cancer.breast_cancer_dataset_utils import get_dataset_breast_cancer
from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.network.ForwardReluTrainer import ForwardReluTrainer

def run(layers):
    # Data
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    train_set, test_set = get_dataset_breast_cancer()

    # Network and Train

    breast_cancer_network = ForwardReLU(layers)
    trainer = ForwardReluTrainer(breast_cancer_network, train_loader=None)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(400)

    print(f"NET: {layer_str}")
    trainer.eval()
    torch.save(breast_cancer_network.state_dict(), f'../../../Networks/breast_cancer/Weights/breast_cancer_net{layer_str}_weights.pth')

if __name__ == '__main__':
    list_layers = [[30, 16, 16, 2],
                   [30, 32, 32, 2],
                   [30, 48, 48, 2],
                   [30, 16, 16, 16, 2],
                   [30, 32, 32, 32, 2],
                   [30, 48, 48, 48, 2],
                   [30, 16, 16, 16, 16, 2],
                   [30, 32, 32, 32, 32, 2],
                   [30, 48, 48, 48, 48, 2]]

    for layers in list_layers:
        run(layers)