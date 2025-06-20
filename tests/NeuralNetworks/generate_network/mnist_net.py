import time

import torch
import torch.optim as optim

from Datasets.mnist.mnist_dataset_utils import get_dataloader_mnist
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.ForwardReluTrainer import ForwardReluTrainer

def run(layers):
    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    print(f"Rodando em: {device}")

    train_loader, test_loader = get_dataloader_mnist()

    model = ForwardReLU(list_len_neurons=layers)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = ForwardReluTrainer(model=model,
                                 train_loader=train_loader,
                                 device=device,
                                 optimizer=optimizer,
                                 test_loader=test_loader)

    trainer.fit(epochs=20)
    trainer.eval()
    torch.save(model.state_dict(), f'../../../Networks/mnist/Weights/mnist_net{layer_str}_weights.pth')

if __name__ == '__main__':
    list_layers = [[28*28, 16, 16, 10],
                   [28*28, 32, 32, 10],
                   [28*28, 48, 48, 10],
                   [28*28, 16, 16, 16, 10],
                   [28*28, 32, 32, 32, 10],
                   [28*28, 48, 48, 48, 10],
                   [28*28, 16, 16, 16, 16, 10],
                   [28*28, 32, 32, 32, 32, 10],
                   [28*28, 48, 48, 48, 48, 10]]

    for layers in list_layers:
        start = time.time()
        run(layers)
        print(f"Treino {layers}", time.time()-start)