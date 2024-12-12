from collections import deque

import docplex.mp.model as mp
import pandas as pd

from src.modeler.milp.Codificator import Codificator
from src.modeler.network.wine.NeuralNetwork import NeuralNetwork


class Explanator:
    def __init__(self, network: NeuralNetwork, dataframe: pd.DataFrame, input_tensors):
        self.network = network
        self.dataframe = dataframe
        self.input_tensors = input_tensors
        dataframe.drop(columns='target', inplace=True)
        self.codificator = Codificator(network, dataframe)
        self.codificator.codify_network_milp_large_bounds()


    def back_explication(self, instance_index):

        bounds = deque()

        bounds.appendleft(self._explain_last_layer(instance_index))
        # for index_layer in self.network.len_layers:

        return bounds


    def _explain_last_layer(self, instance_index):
        index_output_network = self.network.forward(self.input_tensors[instance_index]).argmax().item()
        input_values = self.network.get_layer_values(self.input_tensors[instance_index],
                                                     self.network.len_layers - 2
                                                     ).tolist()
        weights = self.network.layers[-1].weight.detach().numpy()
        bias = self.network.layers[-1].bias.detach().numpy()

        milp = mp.Model()

        input_vars = []
        input_constraints = []

        output_vars = []
        weighted_sum_constraints = []

        for j_neuron, bound in enumerate(self.codificator.bounds_large[-2]):
            lb, ub = bound
            input_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'in_{j_neuron}'))
            input_constraints.append(milp.add_constraint(input_vars[j_neuron] == input_values[j_neuron],
                                                          ctname=f'input_{j_neuron}'
                                                          ))

        for j_neuron, bound in enumerate(self.codificator.bounds_large[-1]):
            lb, ub = bound
            output_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'out_{j_neuron}'))
            weighted_sum_constraints.append(
                milp.add_constraint(output_vars[j_neuron] == weights[j_neuron, :] @ input_vars + bias[j_neuron],
                                    ctname=f'weighted_sum_{j_neuron}'
                                    ))

        bounds = []
        lb_j_plus_e = ub_j_minus_e = float()

        for j in range(len(input_values)):

            # print(f"j: {j}")

            milp.remove_constraint(input_constraints[j])

            if input_vars[j].ub <= 0:
                bounds.append((input_vars[j].lb, input_vars[j].ub))
                milp.add_constraint(input_constraints[j])
                continue

            for o in range(self.network.list_len_neurons[-1]):
                if o == index_output_network:
                    continue
                to_min_constraint = milp.add_constraint(input_values[j] <= input_vars[j])

                # add
                output_constraint = milp.add_constraint( output_vars[o] +0.00001 >= output_vars[index_output_network],
                                                         ctname=f'O_i>=O_j'
                                                         )

                # solve
                milp.minimize(input_vars[j])
                milp.solve()
                milp.remove_objective()

                if milp.solution is None:
                    # print(f"minimize is none: o{o}")
                    ub_j_minus_e = input_vars[j].ub
                else:
                    minimized_x_j = milp.solution.get_objective_value()
                    ub_j_minus_e = minimized_x_j - (minimized_x_j - input_values[j]) * 0.01
                    input_vars[j].ub = ub_j_minus_e

                milp.remove_constraint(to_min_constraint)

                milp.maximize(input_vars[j])
                milp.solve()
                milp.remove_objective()
                milp.remove_constraint(output_constraint)

                if milp.solution is None:
                    # print(f"maximize is none: o{o}")
                    lb_j_plus_e = input_vars[j].lb
                    continue
                else:
                    maximized_x_j = milp.solution.get_objective_value()
                    lb_j_plus_e = maximized_x_j + (input_values[j] - maximized_x_j) * 0.01
                    input_vars[j].lb = lb_j_plus_e
            if self.assert_for_these_bounds(milp, index_output_network, output_vars):
                bounds.append((input_values[j], input_values[j]))

                milp.add_constraint(input_constraints[j])
                print('Segunda tentativa: \n')
                self.assert_for_these_bounds(milp, index_output_network, output_vars, raises=1)
                continue
            bounds.append((lb_j_plus_e, ub_j_minus_e))



        self.assert_for_these_bounds(milp, index_output_network, output_vars, raises=1)

        return bounds

    def assert_for_these_bounds(self, milp, index_output_network, output_vars, raises=0):
        for o in range(self.network.list_len_neurons[-1]):
            if o == index_output_network:
                continue

            if raises:
                print(f'iteração: o={o}, predicted={index_output_network}', end='')

            # add
            output_constraint = milp.add_constraint(output_vars[o] >= output_vars[index_output_network], ctname=f'O_i>=O_j')
            milp.solve()
            if milp.solution is not None:
                if raises:
                    print("\nErro: isso não deveria acontecer!\n")
                    print("\nSolution:\n")
                    for var in milp.iter_variables():
                        print(f"{var.name}: {milp.solution.get_value(var)}")
                    print("\n")
                    milp.remove_constraint(output_constraint)
                    raise AssertionError("Milp solved and raises param is true")
                return True
            else:
                if raises:
                    print(" solution is None, OK")
                milp.remove_constraint(output_constraint)
                return False


    def _explain_intermediate_layer(self, index_previous_layer):
        pass

    def _explain_first_layer(self):
        pass