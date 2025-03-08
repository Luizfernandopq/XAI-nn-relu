import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.modeler.milp.Explanator import Explanator
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.SimpleDataset import SimpleDataset

if __name__ == '__main__':

    bunch = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target,
                                                        test_size=0.33, shuffle=False)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)
    train_set.eat_other(test_set)

    print(bunch.data)
    print(bunch.target)
    print()
    print(train_set.to_dataframe())


    explanator = Explanator(network=ForwardReLU([13, 13, 3]),
                            dataset=train_set,
                            dataframe=train_set.to_dataframe())

    print(explanator.dataframe)
    print(explanator.codificator.data)
    print(explanator.codificator.target)
    print(explanator.bounds.input_types)