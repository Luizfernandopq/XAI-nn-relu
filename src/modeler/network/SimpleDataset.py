import pandas as pd
import torch
from dateutil.rrule import rrule
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    X, y devem ser do tipo torch.Tensor
    """
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

    def eat_other(self, other):
        """Incorpora outro SimpleDataset
        Args:
            other (SimpleDataset): dataset para ser incorporado

        Returns:
            self
        """

        self.X = torch.cat([self.X, other.X])
        self.y = torch.cat([self.y, other.y])
        return self

    def to_dataframe(self, target=True):
        X = self.X.view(self.X.size(0), -1).cpu().numpy()
        df = pd.DataFrame(X)
        if target:
            df['target'] = self.y.cpu().numpy()
        return df