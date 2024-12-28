import time

import torch
from torch import nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda import device
from torch.utils.data import DataLoader

from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer

def run():
    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Rodando em: {device}")

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    # Modelo, função de perda e otimizador
    model = ForwardReLU(list_len_neurons=[28 * 28, 128, 20, 10])

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = ForwardReluTrainer(model=model,
                                 train_loader=train_loader,
                                 device=device,
                                 optimizer=optimizer,
                                 test_loader=test_loader)

    trainer.fit(epochs=100)
    trainer.eval()


def grid_search():
    devices = ['cuda', 'cpu']
    for device in devices:
        times = []
        for i in range(2):
            start = time.time()
            run()
            times.append(time.time() - start)

        print("Treinamento completo em:")
        print(f"Device: {device} | tempo: {sum(times)/2}")

if __name__ == '__main__':

    run()