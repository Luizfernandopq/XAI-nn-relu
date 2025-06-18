import pickle
import time

import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset


def run(layers, num_instances=1000):
    # Configurações
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Rodando em: {device}")

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

    # Dataset e DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = trainset.data.numpy()

    y_train = trainset.targets.numpy()
    X_train = X_train.reshape(X_train.shape[0], -1)

    X_test = testset.data.numpy()

    y_test = testset.targets.numpy()
    X_test = X_test.reshape(X_test.shape[0], -1)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)

    mnist_network = ForwardReLU(layers)
    mnist_network.load_state_dict(torch.load(f'../../../Networks/mnist/Weights/mnist_net{layer_str}_weights01.pth',
                                             weights_only=True))


    with open(f"../../../Networks/mnist/Bounds/mnist_bound{layer_str}.pkl", "rb") as f:
        bounds = pickle.load(f)


    weights = [layer.weight.detach().numpy() for layer in mnist_network.layers if hasattr(layer, 'weight')]
    biases = [layer.bias.detach().numpy() for layer in mnist_network.layers if
              hasattr(layer, 'bias') and layer.bias is not None]

    explainer = LimeTabularExplainer(
        training_data=X_train,
        mode="classification",
        feature_names=[i for i in range(784)],
        class_names=[str(i) for i in range(10)],
        discretize_continuous=False
    )
    start = time.time()
    for index, instance in enumerate(X_test[:num_instances]):
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=mnist_network.predict_by_numpy,
            num_features=500
        )

        # print(exp.as_list())
        # print(len(exp.as_list()))

        # explication = np.zeros(784, dtype=np.float32)

        # instance = scaler.inverse_transform(instance.reshape(1, -1))
        # image = instance.reshape(28, 28)
        # plt.imshow(image, cmap='gray')
        # # plt.title(f'Label: {y_true}')
        # plt.title(f'Instance: {y_test[index]}')
        #
        # plt.axis('off')
        # plt.show(block=False)  # Mostra a imagem sem bloquear a execução
        # plt.pause(5)  # Espera 2 segundos
        # # plt.close()  # Fecha a janela da imagem
        #
        # for pixel, _ in exp.as_list():
        #     explication[pixel] = 1.0
        #
        # # PLOT
        # image2 = explication.reshape(28, 28)
        # plt.imshow(image2, cmap='RdGy')
        # plt.title(f'Y_TRUE: {y_test[index]}')
        # plt.axis('off')
        # plt.show(block=False)
        # plt.pause(4)
        # plt.close()

    print("Tempo total:", time.time() - start)

    print("Instancias:", num_instances)


if __name__ == '__main__':
    list_layers = [[28 * 28, 16, 10],
                   # [28 * 28, 16, 16, 10],
                   # [28 * 28, 32, 10],
                   [28 * 28, 32, 32, 10]#,
                   ]#[28 * 28, 64, 10]]
    for layers in list_layers:
        print(f"Rodando: {layers}")
        run(layers)
