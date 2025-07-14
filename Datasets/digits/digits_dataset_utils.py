import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.relax_explainer.network.SimpleDataset import SimpleDataset


def get_dataset_digits():
    bunch = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target,
                                                        test_size=0.3,
                                                        random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)
    return train_set, test_set