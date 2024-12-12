import pandas as pd

from sklearn.datasets import load_wine

from src.modeler.milp.Explanator import Explanator
from src.modeler.network.wine.NeuralNetwork import NeuralNetwork
from src.modeler.network.wine.Trainer import Trainer

if __name__ == '__main__':

    # dataset

    bunch = load_wine()

    dataframe = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    dataframe['target'] = bunch.target

    # Neural Network
    nn_model = NeuralNetwork(len_layers=3, list_len_neurons=[13, 13, 3])

    # Trainer

    trainer = Trainer(bunch.data, bunch.target)
    nn_model = trainer.default_train(nn_model)
    trainer.eval()

    # Explanator

    explanator = Explanator(nn_model, dataframe, trainer.X_test_t)
    # bounds = explanator.back_explication(0)

    # Explain all test

    for index_instance in range(len(trainer.X_test_t)):
        print("instância: ", index_instance)
        explanator.back_explication(index_instance)
