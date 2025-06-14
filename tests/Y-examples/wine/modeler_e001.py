import pandas as pd
from sklearn.datasets import load_wine

from src.modeler.milp.Codificator import Codificator
from src.modeler.network.wine.NeuralNetwork import NeuralNetwork
from src.modeler.network.wine.Trainer import Trainer

"""
Teste para encontrar os bounds de uma rede neural simples 13x13x3
"""

if __name__ == '__main__':
    # dataset

    bunch = load_wine()

    # Neural Network
    nn_model = NeuralNetwork(len_layers=3, list_len_neurons=[13, 13, 3])

    # Trainer

    trainer = Trainer(bunch.data, bunch.target)
    nn_model = trainer.default_train(nn_model)
    trainer.eval()

    # Codificator

    dataframe = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    # dataframe['target'] = bunch.target
    codificator = Codificator(nn_model, dataframe)

    codificator.codify_network_find_bounds()

    for i, bounds in enumerate(codificator.bounds.layers):
        print(i, bounds)