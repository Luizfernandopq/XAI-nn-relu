from docplex.mp.model import Model
from cplex import infinity
import numpy as np
import tensorflow as tf
import pandas as pd


class NN_milp:
    def __init__(self, network, dataframe):
        self.network = network
        self.dataframe = dataframe
        self.milp_repr = None
        self.input_bounds = []
        self.input_variables = []
        self.intermediate_variables = []
        self.decision_variables = []
        self.output_variables = None

    def codify_milp_network(self):
        self.milp_repr = Model()
        self.get_input_variables_and_bounds()

        for idx_layer, layer in enumerate(self.network.layers):
            len_neurons = layer.get_weights()[0].shape[1]
            if layer == self.network.layers[-1]:
                self.output_variables = self.milp_repr.continuous_var_list(len_neurons, lb=-infinity, name='o')
                break

            self.intermediate_variables.append(self.milp_repr.continuous_var_list(len_neurons,
                                                                                  lb=0,
                                                                                  name='y',
                                                                                  key_format=f"_{idx_layer}_%s"))

            self.decision_variables.append(self.milp_repr.binary_var_list(len_neurons,
                                                                          name='a',
                                                                          lb=0,
                                                                          ub=1,
                                                                          key_format=f"_{idx_layer}_%s"))

    def get_input_variables_and_bounds(self):
        for column_index, column in enumerate(self.dataframe.columns):
            unique_values = self.dataframe[column].unique()
            lower_bound, upper_bound = unique_values.min(), unique_values.max()
            name = f'x_{column_index}'
            if len(unique_values) == 2:
                self.input_variables.append(self.milp_repr.binary_var(name=name))
            elif np.any(unique_values.astype('int64') != unique_values.astype('float64')):
                self.input_variables.append(self.milp_repr.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
            else:
                self.input_variables.append(self.milp_repr.integer_var(lb=lower_bound, ub=upper_bound, name=name))
            self.input_bounds.append((lower_bound, upper_bound))
        return self.input_variables, self.input_bounds

    def insert_output_constraints(self, idx_network_output_argmax, binary_variables):
        pass
