
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from Datasets.mnist.mnist_dataset_utils import get_dataloader_mnist
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.ForwardReluTrainer import ForwardReluTrainer

if __name__ == '__main__':
    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    train_loader, test_loader = get_dataloader_mnist()

    model = ForwardReLU(list_len_neurons=[28 * 28, 16, 16, 10])

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = ForwardReluTrainer(model=model,
                                 train_loader=train_loader,
                                 device=device,
                                 optimizer=optimizer,
                                 test_loader=test_loader)

    trainer.fit(epochs=20)
    trainer.eval()
    # torch.save(model.state_dict(), f'../../../Networks/N-Weights/mnist_net_784x16x16x10_weights01.pth')
