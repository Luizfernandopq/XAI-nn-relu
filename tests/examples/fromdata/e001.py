import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.modeler.network.SimpleDataset import SimpleDataset

if __name__ == '__main__':

    bunch = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target,
                                                        test_size=0.33,
                                                        random_state=42)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)
    train_set.eat_other(test_set)
    # for X, y in train_set:
    #      print(X, y)

