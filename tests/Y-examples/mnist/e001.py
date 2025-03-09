import torch
from torch import nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from src.modeler.network.mnist.MnistNN import MnistNN
from src.modeler.network.mnist.Trainer import Trainer

if __name__ == '__main__':

    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Rodando em: {device}")

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(testset, batch_size=64, shuffle=False)

    # Modelo, função de perda e otimizador
    model = MnistNN(len_layers= 4,list_len_neurons=[28*28, 64, 20, 10])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader
    )

    trainer.fit(epochs=10)
    trainer.eval()


