import time
import docplex.mp.model as mp
import numpy as np
import pandas as pd

from src.modeler.milp.Codificator import Codificator
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.SimpleDataset import SimpleDataset


class Explanator:
    def __init__(self, network: ForwardReLU, dataset: SimpleDataset, dataframe=None):
        self.network = network
        self.dataset = dataset

        if dataframe is None:
            dataframe = dataset.to_dataframe()
        data = dataframe.drop("target", axis=1)

        self.codificator = Codificator(network, data)

        self.bounds = self.codificator.codify_network_find_bounds()


    def back_explication(self, instance_index):
        # Prep
        input, y_true = self.dataset[instance_index]
        neuron_values = self.network.get_all_neuron_values(input)
        out_index = np.argmax(neuron_values[-1])
        bounds_explained = []

        # Run
        bounds_explained.insert(0, self._explain_output_layer(out_index, neuron_values[-2]))

        for index_layer in range(self.network.len_layers - 2, 1, -1):
            print(f"explain intermediate: {index_layer}")

        self._explain_input_layer(bounds_explained[0], neuron_values[0])
        return bounds_explained

    def _explain_output_layer(self, out_pred_index, input_values):
        weights = self.network.layers[-1].weight.detach().numpy()
        bias = self.network.layers[-1].bias.detach().numpy()

        milp = mp.Model()
        milp.parameters.simplex.tolerances.feasibility.set(1e-9)

        input_vars = []
        output_vars = []
        input_constraints = []
        weighted_sum_constraints = []
        for j_neuron, bound in enumerate(self.bounds.layers[-2]):
            lb, ub = self._relu_bounds(bound)
            input_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'in_{j_neuron}'))
            input_constraints.append(milp.add_constraint(input_vars[j_neuron] == input_values[j_neuron],
                                                          ctname=f'input_{j_neuron}'
                                                          ))

        for j_neuron, bound in enumerate(self.bounds.layers[-1]):
            lb, ub = bound
            output_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'out_{j_neuron}'))
            weighted_sum_constraints.append(
                milp.add_constraint(output_vars[j_neuron] == weights[j_neuron, :] @ input_vars + bias[j_neuron],
                                    ctname=f'weighted_sum_{j_neuron}'
                                    ))
        bounds = []
        for j in range(len(input_values)):
            milp.remove_constraint(input_constraints[j])
            if input_vars[j].ub <= 0:
                lb, ub = self.bounds.layers[-2][j]
                bounds.append((lb, ub))
                milp.add_constraint(input_constraints[j])
                continue

            for o in range(self.network.list_len_neurons[-1]):
                if o == out_pred_index:
                    continue

                output_constraint = milp.add_constraint( output_vars[o] >= output_vars[out_pred_index],
                                                         ctname=f'O_i>=O_j'
                                                         )

                self._find_interval(milp, input_value=input_values[j], input_var=input_vars[j])

                milp.remove_constraint(output_constraint)

            if input_vars[j].lb >= input_values[j] *0.9999 > 0 or input_vars[j].ub <= input_values[j]*1.0001:
                milp.add_constraint(input_constraints[j])

                # AS linhas comentadas são uma tentativa desesperada de explicar a camada inicial
                # bounds.append((input_values[j], input_values[j]))
                # continue

            if input_vars[j].lb == 0:
                bounds.append((self.bounds.layers[-2][j][0], input_vars[j].ub))
                continue

            bounds.append((input_vars[j].lb, input_vars[j].ub))

        # Essa função causa overhead
        self.assert_for_these_bounds(milp, out_pred_index, output_vars, raises=1)

        return bounds

    def assert_for_these_bounds(self, milp, index_output_network, output_vars, raises=0):
        for o in range(self.network.list_len_neurons[-1]):
            if o == index_output_network:
                continue
            # add
            output_constraint = milp.add_constraint(output_vars[index_output_network] <= output_vars[o], ctname=f'O_i>=O_j')
            milp.solve()
            milp.remove_constraint(output_constraint)
            if milp.solution is not None:
                if raises:
                    print("\nErro: isso não deveria acontecer!\n")
                    print("\nSolution:\n")
                    for var in milp.iter_variables():
                        print(f"{var.name}: {milp.solution.get_value(var)}")
                    print("\n")
                    raise AssertionError("Milp solved and raises param is true")
                return False
        return True


    def _explain_intermediate_layer(self, index_previous_layer):
        pass

    def _explain_input_layer(self, next_layer_bounds, input_values):
        weights = self.network.layers[0].weight.detach().numpy()
        bias = self.network.layers[0].bias.detach().numpy()

        milp = mp.Model()
        milp.parameters.simplex.tolerances.feasibility.set(1e-8)

        input_vars = []
        output_vars = []
        input_constraints = []
        weighted_sum_constraints = []
        for j_neuron, bound in enumerate(self.bounds.layers[0]):
            lb, ub = bound
            input_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'in_{j_neuron}'))
            input_constraints.append(milp.add_constraint(input_vars[j_neuron] == input_values[j_neuron],
                                                         ctname=f'input_{j_neuron}'
                                                         ))

        for j_neuron, bound in enumerate(self.bounds.layers[1]):
            lb, ub = bound
            output_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'out_{j_neuron}'))
            weighted_sum_constraints.append(
                milp.add_constraint(output_vars[j_neuron] == weights[j_neuron, :] @ input_vars + bias[j_neuron],
                                    ctname=f'weighted_sum_{j_neuron}'
                                    ))

        nones = [0, 0]
        for j in range(len(input_values)):
            milp.remove_constraint(input_constraints[j])
            # print(j, input_vars[j].ub)
            to_min_constraint = milp.add_constraint(input_vars[j] >= input_values[j])
            milp.minimize(input_vars[j])
            for j_next in range(len(next_layer_bounds)):
                lb, ub = next_layer_bounds[j_next]
                y_ge_ub_constraint = milp.add_constraint(output_vars[j_next] >= ub, ctname="y_j >= ub'_j")
                milp.solve()
                if milp.solution is not None:
                    nones[0] += 1
                    minimized_x_j = milp.solution.get_objective_value()
                    ub_j_minus_e = minimized_x_j - (minimized_x_j - input_values[j]) * 0.01
                    input_vars[j].ub = ub_j_minus_e
                else:
                    nones[1] += 1
                milp.remove_constraint(y_ge_ub_constraint)
                y_le_lb_constraint = milp.add_constraint(output_vars[j_next] <= lb, ctname="y_j <= lb'_j")

                milp.solve()
                if milp.solution is not None:
                    nones[0] += 1
                    minimized_x_j = milp.solution.get_objective_value()
                    ub_j_minus_e = minimized_x_j - (minimized_x_j - input_values[j]) * 0.01
                    input_vars[j].ub = ub_j_minus_e
                else:
                    nones[1] += 1
                milp.remove_constraint(y_le_lb_constraint)
        print(f"Sol None = {nones[1]}")
        print(f"Sol True = {nones[0]}")

    def _relu_bounds(self, bound):
        lb, ub = bound
        return max(0, lb), max(0, ub)

    def insert_constraints(self, milp, input_values, weights, bias, layer):
        input_vars = []
        input_constraints = []
        output_vars = []
        for j_neuron, bound in enumerate(self.bounds.layers[layer]):
            if layer != 0:
                bound = self._relu_bounds(bound)
            lb, ub = bound
            input_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'in_{j_neuron}'))
            input_constraints.append(milp.add_constraint(input_vars[j_neuron] == input_values[j_neuron],
                                                         ctname=f'input_{j_neuron}'
                                                         ))

        for j_neuron, bound in enumerate(self.bounds.layers[layer+1]):
            lb, ub = bound
            output_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'out_{j_neuron}'))
            milp.add_constraint(output_vars[j_neuron] == weights[j_neuron, :] @ input_vars + bias[j_neuron],
                                ctname=f'weighted_sum_{j_neuron}'
                                )
        return input_vars, input_constraints, output_vars

    def _find_interval(self, milp, input_value, input_var, is_relu_layer=True):
        to_min_constraint = milp.add_constraint(input_var >= input_value)

        milp.minimize(input_var)
        milp.solve()
        milp.remove_objective()

        if milp.solution is not None:
            minimized_x_j = milp.solution.get_objective_value()
            ub_j_minus_e = minimized_x_j - (minimized_x_j - input_value) * 0.01
            input_var.ub = ub_j_minus_e

        milp.remove_constraint(to_min_constraint)

        if input_value == 0 and is_relu_layer:
            return

        milp.maximize(input_var)
        milp.solve()
        milp.remove_objective()
        if milp.solution is not None:
            maximized_x_j = milp.solution.get_objective_value()
            lb_j_plus_e = maximized_x_j + (input_value - maximized_x_j) * 0.01
            input_var.lb = lb_j_plus_e

