from Datasets.mnist.mnist_dataset_utils import get_dataloader_mnist, get_dataframe_mnist

if __name__ == '__main__':
    df = get_dataframe_mnist()
    print(df.columns)
    for index, instance in df.iterrows():
        print(index, max(instance))
        break

    train, test = get_dataloader_mnist()
    for x, y in train:
        print(max(x[0]))
        break