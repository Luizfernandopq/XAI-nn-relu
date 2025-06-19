
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

from src.legacy.codify_network import codify_network
from src.back_explainer.milp.Codificator import Codificator
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

if __name__ == '__main__':

    # Data
    bunch = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target,
                                                        test_size=0.33,
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

    # Network

    layers = [13, 16, 3]

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    wine_network = ForwardReLU(layers)
    wine_network.load_state_dict(torch.load(f'../../../Networks/wine/Weights/wine_net{layer_str}_weights01.pth',
                                            weights_only=True))

    wine_network.eval()

    # weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    # biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
    #           hasattr(layer, 'bias') and layer.bias is not None]

    mdl , out_bounds = codify_network(wine_network,
                                      train_set.eat_other(test_set).to_dataframe(target=False))

    print(out_bounds)

    mdl_relax, out_bounds_relax = relaxed_codify_network(wine_network,
                                                         train_set.eat_other(test_set).to_dataframe(target=False),
                                                         relax_density=0.25)

    print(out_bounds_relax)

