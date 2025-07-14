import pandas as pd

from Datasets.mnist.mnist_dataset_utils import get_dataloader_mnist_binary

if __name__ == '__main__':
   train_set, test_set = get_dataloader_mnist_binary()
