import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from torch import optim

from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer
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

    wine_model = ForwardReLU([13, 13, 3])

    trainer = ForwardReluTrainer(wine_model,
                                 train_loader=None)

    trainer.update_loaders(train_set, test_set)

    trainer.fit(250)

    trainer.eval()