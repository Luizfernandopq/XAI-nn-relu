import docplex.mp.model as mp
from cplex import infinity
import numpy as np
import tensorflow as tf
import pandas as pd
from main_code.rede_em_milp import slice_bounds as sb

from main_code.rede_em_milp import tjeng
from main_code.rede_em_milp import fischetti


class NN_milp:
    def __init__(self, network, dataframe):
        self.network = network
        self.dataframe = dataframe
        self.milp_repr = None
        self.type_domain_input = None
        self.bounds_input = None
        self.input_variables = []
        self.intermediate_variables = []
        self.decision_variables = []
        self.output_variables = None

    def codify_milp_network(self):
        self.milp_repr = mp.Model()
        self.get_domain_and_bounds_inputs()

        self.get_input_variables()

        for i in range(len(self.network.layers) - 1):
            weights = self.network.layers[i].get_weights()[0]

            self.intermediate_variables.append(
                self.milp_repr.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))

            self.decision_variables.append(
                self.milp_repr.binary_var_list(weights.shape[1],
                                               name='a',
                                               lb=0,
                                               ub=1,
                                               key_format=f"_{i}_%s")
            )

        self.output_variables = self.milp_repr.continuous_var_list(self.network.layers[-1].get_weights()[0].shape[1],
                                                                   lb=-infinity,
                                                                   name='o'
                                                                   )




    def get_input_variables(self):
        pass

    def get_domain_and_bounds_inputs(self):
        pass

    def insert_output_constraints(self, network_output_argmax):
        pass