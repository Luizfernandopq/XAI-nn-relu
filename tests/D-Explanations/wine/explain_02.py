import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.modeler.explainer import generate_explanation
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.ForwardReluTrainer import ForwardReluTrainer
from src.modeler.network.SimpleDataset import SimpleDataset

def save_model(i):
    torch.save(wine_network.state_dict(), f'wine_network_weights{i:02}.pth')

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
    trainer = ForwardReluTrainer(wine_network, train_loader=None)
    trainer.update_loaders(train_set, test_set)
    trainer.fit(1000)

    weights = [layer.weight.detach().numpy() for layer in wine_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in wine_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]
    print(weights)
    save_model(2)
    # for i in range(train_set.__len__()):
    #     print(f"Explain {i}")
    #     instance = train_set.__getitem__(i)
    #     values = wine_network.get_all_neuron_values(instance)
    #     generate_explanation(
    #         np.argmax(values[-1]),
    #
    #     )
    #     print(f"Explain {i}")

