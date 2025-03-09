import pandas as pd
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sympy.matrices.expressions.blockmatrix import bounds
from torch.utils.data import DataLoader

from src.modeler.milp.Explanator import Explanator
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer
from src.modeler.network.SimpleDataset import SimpleDataset

if __name__ == '__main__':

    # Data
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

    # Network and Train

    wine_network = ForwardReLU([13, 3, 3])
    trainer = ForwardReluTrainer(wine_network, train_loader=None)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(1000)

    explanator = Explanator(wine_network,
                            dataset=train_set.eat_other(test_set),
                            dataframe=train_set.to_dataframe())

    for i in range(train_set.__len__()):
        print(f"Explain {i}")
        bounds = explanator.back_explication(i)
        # for j, bound in enumerate(bounds[0]):
            # print(j, explanator.bounds.layers[-2][j])
            # print(j, bound)
        print()