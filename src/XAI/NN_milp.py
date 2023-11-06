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
        self.type_domain_input = []
        self.bounds_input = []
        self.input_variables = []
        self.intermediate_variables = []
        self.decision_variables = []
        self.output_variables = None

    def codify_milp_network(self):
        self.milp_repr = Model()
        self.get_domain_and_bounds_inputs()

        self.get_input_variables()

        self.get_intermediate_variables()

        self.output_variables = self.milp_repr.continuous_var_list(self.network.layers[-1].get_weights()[0].shape[1],
                                                                   lb=-infinity,
                                                                   name='o'
                                                                   )

    def get_input_variables(self):
        for index, (domain, bounds) in enumerate(zip(self.type_domain_input, self.bounds_input)):
            lower_bound, upper_bound = bounds
            name = f'x_{index}'
            if domain == 'C':
                self.input_variables.append(self.milp_repr.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
            elif domain == 'I':
                self.input_variables.append(self.milp_repr.integer_var(lb=lower_bound, ub=upper_bound, name=name))
            else:
                self.input_variables.append(self.milp_repr.binary_var(name=name))

    def get_intermediate_variables(self):
        for idx_layer in range(len(self.network.layers) - 1):
            len_neurons = self.network.layers[idx_layer].get_weights()[0].shape[1]

            self.intermediate_variables.append(self.milp_repr.continuous_var_list(len_neurons,
                                                                                  lb=0,
                                                                                  name='y',
                                                                                  key_format=f"_{idx_layer}_%s"))

            self.decision_variables.append(self.milp_repr.binary_var_list(len_neurons,
                                                                          name='a',
                                                                          lb=0,
                                                                          ub=1,
                                                                          key_format=f"_{idx_layer}_%s"))

    def get_domain_and_bounds_inputs(self):
        columns = self.dataframe.columns
        for column in columns:
            unique_values = self.dataframe[column].unique()
            self.type_domain_input.append(
                'B' if len(unique_values) == 2 else
                'C' if np.any(unique_values.astype(np.int64) != unique_values.astype(np.float64)) else
                'I')
            self.bounds_input.append((self.dataframe[column].min(), self.dataframe[column].max()))

    def insert_output_constraints(self, index_output_predicted, binary_variables):
        pass
