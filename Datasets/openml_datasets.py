import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import openml

from src.relax_explainer.network.SimpleDataset import SimpleDataset


def get_dataset_openml(name: str, test_size=0.20, random_state=42):
    openml_ids = {
        "diabetes": 37,
        "glass": 41,
        "heart-statlog": 53,
        "iris": 61
    }

    if name not in openml_ids:
        raise ValueError(f"Dataset {name} não está na lista suportada.")

    ds = openml.datasets.get_dataset(openml_ids[name])
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)

    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalização Min-Max
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_set = SimpleDataset(X_train_t, y_train_t)
    test_set = SimpleDataset(X_test_t, y_test_t)

    return train_set, test_set
