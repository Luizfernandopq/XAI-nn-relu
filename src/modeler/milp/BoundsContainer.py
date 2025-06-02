
import numpy as np


class BoundsContainer:
    """
            Classe para armazenamento dos bounds de uma codificação

            Args:
                dataframe (pandas.DataFrame)

            Attr:
                input_types (List[str]): uma lista de string ['C', 'B', 'I', ...]

                layers (List[List[(float, float)]]): uma lista de lista de tuplas
                                                     [[(x1_min, x1_max) ...][(y1_1_min, y1_1_max)...]...]
                                                     Lista contém várias listas em que cada lista é uma camada
                                                     e cada item dessa lista é um neurônio
                                                     e a tupla é o min max dos bounds do neurônio
    """

    def __init__(self, dataframe):
        self.layers = []
        self.input_types = []
        self.layers.append(self.get_types_and_bounds(dataframe))

    def __str__(self):
        return self.layers.__str__()

    def get_types_and_bounds(self, dataframe, ignore_int=True):
        columns = dataframe.columns
        bounds = []
        for column in columns:
            unique_values = dataframe[column].unique()
            self.input_types.append(
                'B' if len(unique_values) == 2 else
                'C' if np.any(unique_values.astype(np.int64) != unique_values.astype(np.float64)) or ignore_int else
                'I')
            bounds.append((dataframe[column].min(), dataframe[column].max()))
        return bounds
