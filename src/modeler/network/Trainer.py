import torch
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim

from src.modeler.network.NeuralNetwork import NeuralNetwork


class Trainer:
    def __init__(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.X_train_t = torch.FloatTensor(X_train)
        self.y_train_t = torch.LongTensor(y_train)
        self.X_test_t = torch.FloatTensor(X_test)
        self.y_test_t = torch.LongTensor(y_test)

        self.model = None

    def default_train(self, neural_network: NeuralNetwork):
        self.model = neural_network

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        losses = []
        test_losses = []

        epochs = 1000
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train_t)
            loss = criterion(outputs, self.y_train_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            with torch.no_grad():
                self.model.eval()
                test_outputs = self.model(self.X_test_t)
                test_loss = criterion(test_outputs, self.y_test_t)
                test_losses.append(test_loss.item())
        plt.plot(losses, label="Treinamento")
        plt.plot(test_losses, label="Teste")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Gráfico das Losses de Treinamento e Teste")
        plt.savefig("figures/treino.png")
        return self.model

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            test_outputs = self.model(self.X_test_t)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == self.y_test_t).sum().item() / len(self.y_test_t)
            print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')
            self.confusion_matrix(predicted)

    def confusion_matrix(self, predicted):
        y_true = self.y_test_t.numpy()
        y_pred = predicted.numpy()

        confusion = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Valores Preditos")
        plt.ylabel("Valores Reais")
        plt.title("Matriz de Confusão")
        plt.savefig("figures/cm.png")