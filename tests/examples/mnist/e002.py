import torch
from torch import nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from src.modeler.milp.Explanator import Explanator
from src.modeler.network.mnist.MnistNN import MnistNN
from src.modeler.network.mnist.Trainer import Trainer
import pandas as pd
import torch

def extract_data_and_targets(data_loader):
    """
    Extrai os dados (inputs) e targets de um DataLoader e os organiza em um DataFrame.

    Args:
    - data_loader (DataLoader): O DataLoader contendo os dados.

    Returns:
    - data_df (pd.DataFrame): Um dataframe contendo os dados (inputs) e seus targets.
    """
    all_inputs = []
    all_tensors = []
    all_targets = []

    for inputs, targets in data_loader:
        # Achata as imagens para facilitar a organização em colunas
        batch_size = inputs.size(0)
        flattened_inputs = inputs.view(batch_size, -1)  # (batch_size, feature_count)

        all_inputs.append(flattened_inputs)
        all_tensors.append(inputs)
        all_targets.append(targets)

    # Concatena os batches em um único tensor
    inputs_tensor = torch.cat(all_tensors, dim=0)
    inputs_tensor_aux = torch.cat(all_inputs, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # Converte para numpy para criar o DataFrame
    inputs_array = inputs_tensor_aux.numpy()
    targets_array = targets_tensor.numpy()

    # Cria o DataFrame com os dados e os alvos
    data_df = pd.DataFrame(inputs_array)
    data_df['target'] = targets_array

    return data_df, inputs_tensor

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

    trainer.fit(epochs=1)
    trainer.eval()

    data, tensors = extract_data_and_targets(trainer.val_loader)

    explanator = Explanator(model, data, tensors)
    # bounds = explanator.back_explication(0)

    # Explain all test

    for index_instance in range(150):
        print("instância: ", index_instance)
        explanator.back_explication(index_instance)
