import pickle
import time

import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset


def run(layers):
    if layers is None:
        layers = [13, 16, 3]

    layer_str = "_"
    for i in layers[:-1]:
        layer_str += str(i) + "x"
    layer_str += str(layers[-1])

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

    # Network and Train

    wine_network = ForwardReLU(layers)
    wine_network.load_state_dict(torch.load(f'../../../Networks/wine/Weights/wine_net{layer_str}_weights01.pth',
                                            weights_only=True))

    wine_network.eval()


    explainer = LimeTabularExplainer(
        training_data=X_train,
        mode="classification",
        feature_names=[i for i in range(784)],
        class_names=[str(i) for i in range(10)],
        discretize_continuous=False
    )
    start = time.time()
    for index, instance in enumerate(X_test):
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=wine_network.predict_by_numpy,
            num_features=7
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

    print("Instancias:", len(X_test))


if __name__ == '__main__':
    list_layers = [[13, 16, 3],
                   [13, 16, 16, 3],
                   [13, 32, 3],
                   [13, 32, 32, 3]]
    for layers in list_layers:
        print(f"Rodando: {layers}")
        run(layers)