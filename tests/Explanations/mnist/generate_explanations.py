import pickle
import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.back_explainer.milp.explainer import generate_explanation
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset


def run(layers, num_features=1000):
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

    print("Novo: ")
    tamanhos = []
    ok = 0
    notok = 0
    start = time.time()
    for i in range(num_features):
        instance, y_test = test_set[i]
        values = mnist_network.get_all_neuron_values(instance)
        result, importancias = generate_explanation(
            np.argmax(values[-1]),
            weights,
            biases,
            values,
            bounds
        )
        # print(f"Values: {values[0].tolist()}")
        # print(f"Bounds: {bounds[0]}")
        # print(f"Result: {result[0]}")

        tamanhos.append(len(importancias))
        # print(len(importancias))

        # explication = np.zeros(784, dtype=np.float32)
        # for j in importancias:
        #     explication[j] = 1.0
        #     # instance[j] = np.random.rand()
        #
        # # for j in range(784):
        # #     if bounds[0][j][0] != result[0][j][0] or bounds[0][j][1] != result[0][j][1]:
        # #         explication[j] = 1.0
        #
        # new_values = mnist_network.get_all_neuron_values(instance)
        #
        # # if np.argmax(values[-1]) == np.argmax(new_values[-1]):
        # #     ok+=1
        # #     # continue
        # # notok+=1
        #
        # # PLOT
        # instance = scaler.inverse_transform(instance.reshape(1, -1))
        # image = instance.reshape(28, 28)
        # plt.imshow(image, cmap='gray')
        # # plt.title(f'Label: {y_true}')
        # plt.title(f'Instance: {i} | Y_TRUE: {y_true} | Pred: {np.argmax(values[-1])}')
        #
        # plt.axis('off')
        # plt.show(block=False)  # Mostra a imagem sem bloquear a execução
        # plt.pause(5)  # Espera 2 segundos
        # # plt.close()  # Fecha a janela da imagem
        #
        # # PLOT
        # image2 = explication.reshape(28, 28)
        # plt.imshow(image2, cmap='RdGy')
        # plt.title(f'Y_TRUE: {y_true} | Predicted: {np.argmax(new_values[-1])}')
        # plt.axis('off')
        # plt.show(block=False)
        # plt.pause(4)
        # plt.close()



    print("Tempo total:", time.time() - start)
    print("OK", ok)
    print("NOTOK", notok)

    media = np.mean(tamanhos)
    mediana = np.median(tamanhos)
    maximo = np.max(tamanhos)
    minimo = np.min(tamanhos)

    print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")


    # ------------------Legacy----------------
    # print("Legacy")
    # model, bounds_legacy = codify_network(mnist_network, train_set.eat_other(test_set).to_dataframe(target=False))
    #
    #
    # df = test_set.to_dataframe()
    # len_inputs = []
    # notok=0
    # ok=0
    # start = time.time()
    # for index, instance in df.iterrows():
    #     if index > 4:
    #         break
    #     # print("Legacy explicando:", index)
    #     target = instance["target"].astype(int)
    #     instance = instance.drop("target")
    #     # print(instance)
    #     vinst, _ = test_set[index]
    #     values = mnist_network.get_all_neuron_values(vinst)
    #     # if target == np.argmax(values[-1]):
    #     #     continue
    #
    #     inputs = get_miminal_explanation(model, instance, target, bounds_legacy, 10)
    #     len_inputs.append(len(inputs))
    #
    #     # PLOT --------------------
    #     # explication = np.zeros(784, dtype=np.float32)
    #
    #     indexes = []
    #     for j in inputs:
    #         index_input = int(j.name.split("input")[1]) - 1
    #         # explication[index_input] = 1.0
    #         indexes.append(index_input)
    #
    #     for j in range(len(vinst)):
    #         if j not in indexes:
    #             vinst[j] = np.random.rand()
    #
    #     new_values = mnist_network.get_all_neuron_values(vinst)
    #     if np.argmax(values[-1]) == np.argmax(new_values[-1]):
    #         ok+=1
    #         continue
    #     notok+=1
    #
    #     # image = instance.to_numpy().reshape(28, 28)
    #     # plt.imshow(image, cmap='gray')
    #     # # plt.title(f'Label: {target}')
    #     # plt.title(f'Instance: {index} | Y_TRUE: {target} | Predicted: {np.argmax(values[-1])}')
    #     #
    #     # plt.axis('off')
    #     # plt.show(block=False)  # Mostra a imagem sem bloquear a execução
    #     # plt.pause(4)  # Espera 2 segundos
    #     # # plt.close()  # Fecha a janela da imagem
    #     #
    #     # image2 = explication.reshape(28, 28)
    #     # plt.imshow(image2, cmap='RdGy')
    #     # plt.title(f'Explicação: {target}')
    #     # plt.axis('off')
    #     # plt.show(block=False)  # Mostra a imagem sem bloquear a execução
    #     # plt.pause(4)  # Espera 2 segundos
    #     # plt.close()  # Fecha a janela da imagem
    #
    # print("Tempo total:", time.time() - start)
    #
    # print("OK", ok)
    # print("NOTOK", notok)
    #
    # media = np.mean(len_inputs)
    # mediana = np.median(len_inputs)
    # maximo = np.max(len_inputs)
    # minimo = np.min(len_inputs)
    #
    # print(f"Média: {media}, Mediana: {mediana}, Máximo: {maximo}, Mínimo: {minimo}")


if __name__ == '__main__':
    list_layers = [[28 * 28, 16, 10],
                   # [28 * 28, 16, 16, 10],
                   # [28 * 28, 32, 10],
                   [28 * 28, 32, 32, 10]#,
                   ]#[28 * 28, 64, 10]]
    for layers in list_layers:
        print(f"Rodando: {layers}")
        run(layers)
