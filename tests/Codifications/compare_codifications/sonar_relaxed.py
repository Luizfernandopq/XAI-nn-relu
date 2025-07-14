
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

from Datasets.sonar.sonar_dataset_utils import get_dataset_sonar
from src.legacy.codify_network import codify_network
from src.relax_explainer.network.ForwardReLU import ForwardReLU
from src.relax_explainer.network.SimpleDataset import SimpleDataset
from src.relax_explainer.relaxed_codify_network import relaxed_codify_network

if __name__ == '__main__':


    train_set, test_set = get_dataset_sonar()
    # Network

    layers = [60, 16, 16, 2]

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    sonar_network = ForwardReLU(layers)
    sonar_network.load_state_dict(torch.load(f'../../../Networks/sonar/Weights/sonar_net{layer_str}_weights.pth',
                                            weights_only=True))

    sonar_network.eval()

    # weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    # biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
    #           hasattr(layer, 'bias') and layer.bias is not None]

    mdl , out_bounds = codify_network(sonar_network,
                                      train_set.eat_other(test_set).to_dataframe(target=False))

    print(out_bounds)

    mdl_relax, out_bounds_relax = relaxed_codify_network(sonar_network,
                                                         train_set.eat_other(test_set).to_dataframe(target=False),
                                                         relax_quatity=2)

    print(out_bounds_relax)

    mdl_relax, out_bounds_relax = relaxed_codify_network(sonar_network,
                                                         train_set.eat_other(test_set).to_dataframe(target=False),
                                                         relax_quatity=4)

    print(out_bounds_relax)

    mdl_relax, out_bounds_relax = relaxed_codify_network(sonar_network,
                                                         train_set.eat_other(test_set).to_dataframe(target=False),
                                                         relax_quatity=8)

    print(out_bounds_relax)
