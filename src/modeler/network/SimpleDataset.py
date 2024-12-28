from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        # Retorna o tamanho do conjunto de dados
        return len(self.X)

    def __getitem__(self, idx):
        # Retorna uma amostra e seu rótulo
        x_sample = self.X[idx]
        y_label = self.y[idx]
        return x_sample, y_label