import time

import numpy as np
import pandas as pd
from cplex import infinity

import docplex.mp.model as mp

from src.modeler.milp.BoundsContainer import BoundsContainer
from src.modeler.network.ForwardReLU import ForwardReLU


class Codificator:
    """
            Classe (codificação) responsável por encontrar bounds válidos para todos os neurônios com base nos valores presentes
            no dataset para a entrada da rede

            Args:
                network (ForwardReLU): A rede neural utilizada.
                dataframe (pandas.DataFrame): O dataframe contendo os dados associados.

            Attr:
                network (ForwardReLU): A rede neural que fornecerá pesos, bias e cálculos de valores de neurônios.

                dataframe (pandas.DataFrame): Os dados utilizados para encontrar bounds.

                milp_representation (mp.Model): Representação MILP (Mixed-Integer Linear Programming) da codificação.

                input_variables (List[mp.Var]): Variáveis de entrada para representar os neurônios da camada de entrada.

                intermediate_variables (List[List[mp.Var]]): Variáveis intermediárias geradas como continuous var para
                                                             representar os neurônios das camadas intermediárias.

                decision_variables (List[List[mp.Var]]): Variáveis de decisão no contexto MILP, geradas como binárias
                                                         para representar a ativação ReLU.

                output_variables (List[mp.Var]): Variáveis de saída para representar os neurônios da camada de saída
    """

    def __init__(self, network: ForwardReLU, dataframe: pd.DataFrame):
        self.network = network
        self.data = dataframe
        self.bounds = BoundsContainer(self.data)
        self.milp_represetation = None

        self.input_variables = None
        self.intermediate_variables = None
        self.decision_variables = None
        self.output_variables = None

    def codify_network_find_bounds(self):
        self.milp_represetation = mp.Model()

        self._init_input_variables()

        self._init_intermediate_variables()

        self.output_variables = self.milp_represetation.continuous_var_list(self.network.layers[-1].weight.detach().numpy().T.shape[1],
                                                                            lb=-infinity,
                                                                            name='o'
                                                                            )

        bounds = self._codify_tjeng()

        return bounds.layers

    def _codify_tjeng(self):


        len_layers = len(self.network.layers)
        for i in range(len_layers):
            A = self.network.layers[i].weight.detach().numpy()

            b = self.network.layers[i].bias.detach().numpy()

            x = self.input_variables if i == 0 else self.intermediate_variables[i - 1]

            bounds = []

            if i != (len_layers - 1):
                a = self.decision_variables[i]
                y = self.intermediate_variables[i]
            else:
                y = self.output_variables

            for j in range(A.shape[0]):
                weighted_sum = A[j, :] @ x + b[j]

                ub = self._maximize(weighted_sum)
                lb = self._minimize(weighted_sum)

                if i != len_layers - 1:
                    if ub <= 0:
                        self.milp_represetation.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')

                    elif lb >= 0:
                        self.milp_represetation.add_constraint(weighted_sum == y[j], ctname=f'c_{i}_{j}')
                    else:
                        self.milp_represetation.add_constraint(y[j] <= weighted_sum - lb * (1 - a[j]))
                        self.milp_represetation.add_constraint(y[j] >= weighted_sum)
                        self.milp_represetation.add_constraint(y[j] <= ub * a[j])

                    bounds.append((lb, ub))

                else:
                    self.milp_represetation.add_constraint(weighted_sum == y[j])

                    bounds.append((lb, ub))
            self.bounds.layers.append(bounds)
        return self.bounds

    def _maximize(self, expr):
        self.milp_represetation.maximize(expr)
        self.milp_represetation.solve()
        if self.milp_represetation.solution is None:
            return 0
        ub = self.milp_represetation.solution.get_objective_value()
        self.milp_represetation.remove_objective()
        return ub

    def _minimize(self, expr):
        self.milp_represetation.minimize(expr)
        self.milp_represetation.solve()
        if self.milp_represetation.solution is None:
            return 0
        lb = self.milp_represetation.solution.get_objective_value()
        self.milp_represetation.remove_objective()
        return lb

    def _init_input_variables(self):
        self.input_variables = []
        for index, (domain_type, bounds) in enumerate(zip(self.bounds.input_types, self.bounds.layers[0])):
            lower_bound, upper_bound = bounds
            name = f'x_{index}'
            if domain_type == 'C':
                self.input_variables.append(self.milp_represetation.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
            elif domain_type == 'I':
                self.input_variables.append(self.milp_represetation.integer_var(lb=lower_bound, ub=upper_bound, name=name))
            else:
                self.input_variables.append(self.milp_represetation.binary_var(name=name))

    def _init_intermediate_variables(self):
        self.intermediate_variables = []
        self.decision_variables = []
        for idx_layer in range(len(self.network.layers) - 1):
            len_neurons = self.network.layers[idx_layer].weight.detach().numpy().T.shape[1]

            self.intermediate_variables.append(self.milp_represetation.continuous_var_list(len_neurons,
                                                                                  lb=0,
                                                                                  name='y',
                                                                                  key_format=f"_{idx_layer}_%s"))

            self.decision_variables.append(self.milp_represetation.binary_var_list(len_neurons,
                                                                          name='a',
                                                                          lb=0,
                                                                          ub=1,
                                                                          key_format=f"_{idx_layer}_%s"))
