from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.relax_explainer.network.SimpleDataset import SimpleDataset


def get_dataset_sonar():
    path_atual = Path(__file__).resolve().parent
    df = pd.read_csv(f"{path_atual}/sonar.csv", index_col=0)
    X = df.iloc[:, :-1].astype('float64').values
    y = df.iloc[:, -1].astype('int64').values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
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