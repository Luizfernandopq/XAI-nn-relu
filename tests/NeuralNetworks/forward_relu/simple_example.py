import torch

from src.back_explainer.network.ForwardReLU import ForwardReLU

if __name__ == '__main__':
    model = ForwardReLU([13, 13, 13, 3])

    print("Layers: ", model.layers)
    print("Layer 1: ", model.get_layer_values(torch.randn(1, 13), 1))
    print("Layer 2: ", model.get_layer_values(torch.randn(1, 13), 2))
    print("Layer out: ",model.forward(torch.randn(1, 13)))
