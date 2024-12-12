import torch

class Trainer:
    def __init__(self, model, device, optimizer, criterion, train_loader, val_loader=None):
        """
        Inicializa a classe Trainer.

        Args:
        - model: O modelo PyTorch a ser treinado.
        - device: O dispositivo ('cuda' ou 'cpu').
        - optimizer: O otimizador para atualização dos pesos.
        - criterion: A função de perda.
        - train_loader: DataLoader para os dados de treinamento.
        - val_loader: (Opcional) DataLoader para os dados de validação.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

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
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Calcula a acurácia
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self, epochs):
        """Treina o modelo por múltiplas épocas."""
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

            if self.val_loader:
                val_loss, val_accuracy = self.validate_epoch()
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
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