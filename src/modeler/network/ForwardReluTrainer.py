import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader

from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.SimpleDataset import SimpleDataset


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

        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        if optimizer is None:
            optimizer = optim.Adam(model.parameters())
        self.optimizer = optimizer
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        self.test_loader = test_loader

    def update_loaders(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.train_loader = DataLoader(SimpleDataset(X_train, y_train))
        self.test_loader = DataLoader(SimpleDataset(X_test, y_test))


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
            loss = self.criterion(outputs.to(self.device), labels.to(self.device))

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
                loss = self.criterion(outputs.to(self.device), labels.to(self.device))

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
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

            if self.test_loader:
                val_loss, val_accuracy = self.validate_epoch()
                # print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Contagem de acertos e total de amostras
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Calcula a acurácia
        accuracy = correct / total
        print(f'Acurácia no conjunto avaliado: {accuracy * 100:.2f}%')
        return accuracy