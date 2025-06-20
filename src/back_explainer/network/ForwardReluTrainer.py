import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from torch import optim, nn
from torch.utils.data import DataLoader

from src.back_explainer.network.ForwardReLU import ForwardReLU
from src.back_explainer.network.SimpleDataset import SimpleDataset


class ForwardReluTrainer:
    def __init__(self, model: ForwardReLU, train_loader, device='cpu', optimizer=None, criterion=None, test_loader=None):
        """
        Inicializa a classe Trainer.

        Args:
        - model: O modelo PyTorch a ser treinado.
        - train_loader: DataLoader para os dados de treinamento.
        - device: O dispositivo ('cuda' ou 'cpu').
        - optimizer: O otimizador para atualização dos pesos.
        - criterion: A função de perda.
        - test_loader: (Opcional) DataLoader para os dados de validação.
        """

        self.model = model
        self.train_loader = train_loader
        self.device = device
        if optimizer is None:
            optimizer = optim.Adam(model.parameters())
        self.optimizer = optimizer
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        self.test_loader = test_loader

    def update_loaders(self, train_set, test_set):
        self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


    def train_epoch(self):
        """Treina o modelo por uma única época."""
        self.model.train()
        running_loss = 0.0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zera os gradientes acumulados
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Atualiza os pesos
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        """Valida o modelo por uma única época."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Calcula a acurácia
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.test_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self, epochs):
        """Treina o modelo por múltiplas épocas."""
        self.model.to(self.device)
        losses = []
        test_losses = []

        for epoch in range(epochs):
            train_loss = self.train_epoch()

            losses.append(train_loss)

            if self.test_loader:
                test_loss, test_accuracy = self.validate_epoch()
                test_losses.append(test_loss)
                if epoch % (epochs/10) == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
                    print(f"Validation Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        plt.plot(losses, label="Treinamento")
        plt.plot(test_losses, label="Teste")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Gráfico das Losses de Treinamento e Teste")
        # plt.savefig("figures/treino.png")

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0

        predicteds = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Contagem de acertos e total de amostras
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                predicteds.extend(predicted.cpu().numpy())

        # Calcula a acurácia
        accuracy = correct / total
        print(f'Acurácia no conjunto avaliado: {accuracy * 100:.2f}%')
        self.confusion_matrix(predicteds)
        return accuracy

    def confusion_matrix(self, predicted):
        try:
            y_true = self.test_loader.dataset.y.numpy()
        except:
            y_true = self.test_loader.dataset.targets.numpy()
        y_pred = predicted

        confusion = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Valores Preditos")
        plt.ylabel("Valores Reais")
        plt.title("Matriz de Confusão")
        # plt.savefig("figures/cm.png")