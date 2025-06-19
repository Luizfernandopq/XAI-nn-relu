from pathlib import Path

import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_dataframe_mnist(target=False):
    path_atual = Path(__file__).resolve().parent
    trainset = datasets.MNIST(root=path_atual, train=True, download=True)
    testset = datasets.MNIST(root=path_atual, train=False, download=True)
    X_train = trainset.data.numpy() / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = testset.data.numpy() / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = trainset.targets.numpy()
    y_test = testset.targets.numpy()
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)
    if target:
        df_train['target'] = y_train
        df_test['target'] = y_test
    return pd.concat([df_train, df_test], ignore_index=True)



def get_dataloader_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    path_atual = Path(__file__).resolve().parent
    trainset = datasets.MNIST(root=path_atual, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=path_atual, train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    return train_loader, test_loader