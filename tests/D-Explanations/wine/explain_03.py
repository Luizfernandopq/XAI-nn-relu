
import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.modeler.explainer import generate_explanation
from src.modeler.milp.Codificator import Codificator
from src.modeler.network.ForwardReLU import ForwardReLU
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

    wine_network = ForwardReLU([13, 13, 3])
    wine_network.load_state_dict(torch.load('wine_net_[13, 13, 3]_not_normal_weights01.pth', weights_only=True))

    wine_network.eval()

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]

    codificator = Codificator(wine_network, train_set.eat_other(test_set).to_dataframe(target=False))
    bounds = codificator.codify_network_find_bounds()
    tamanhos = []
    for i in range(train_set.__len__()):
        print(f"Explain: {i}")
        instance, y_true = train_set[i]
        values = wine_network.get_all_neuron_values(instance)
        result = generate_explanation(
            np.argmax(values[-1]),
            weights,
            biases,
            values,
            bounds
        )
        print(f"Values: {values[-3].tolist()}")
        print(f"Bounds: {bounds[-3]}")
        print(f"Result: {result[-2]}")
        tamanho = 0
        for i in range(len(instance)):
            if bounds[-3][i][1] != result[-2][i][1] or bounds[-3][i][0] != result[-2][i][0]:
                tamanho+=1
        tamanhos.append(tamanho)

    media = np.mean(tamanhos)
    mediana = np.median(tamanhos)
    maximo = np.max(tamanhos)
    minimo = np.min(tamanhos)

    print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")
