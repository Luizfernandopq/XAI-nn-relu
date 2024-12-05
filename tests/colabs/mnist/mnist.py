import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
import torch.nn as nn
from tests.colabs.mnist.mnistNN import MnistNN


def evaluate_accuracy(test_loader, model, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples * 100
    return accuracy


if __name__ == '__main__':

    print("Inicializando")

    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(images[i][0], cmap=plt.cm.binary)
    #     plt.xlabel(labels[i].item())
    # plt.show()

    batch_size = 32
    dummy_input = torch.randn(batch_size, 1, 28, 28)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando o dispositivo: {device}")
    model = MnistNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_losses = []
    train_accuracies = []

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct_predictions / total_samples * 100

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acurácia: {epoch_accuracy:.2f}%")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Perda (Loss)', color='tab:red')
    ax1.plot(range(1, num_epochs + 1), train_losses, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Acurácia (%)', color='tab:blue')
    ax2.plot(range(1, num_epochs + 1), train_accuracies, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title("Perda (Loss) e Acurácia durante o Treinamento")
    plt.show()

    model.eval()

    data_iter = iter(testloader)

    # Após o treinamento, calcule a acurácia do conjunto de teste
    test_accuracy = evaluate_accuracy(testloader, model, device)
    print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}%")