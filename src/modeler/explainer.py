import time
import docplex.mp.model as mp
import numpy as np
import pandas as pd

from src.modeler.milp.Codificator import Codificator
from src.modeler.network.ForwardReLU import ForwardReLU
from src.modeler.network.SimpleDataset import SimpleDataset

def generate_explanation(
        output_predicted_index,
        weights,
        bias,
        neuron_values,
        original_bounds
):
    """

    Args:
        output_predicted_index (int): índice da camada de saida dado como output da rede neural

        weights (List[List[List[float]]]): Lista contendo matrizes [camada1, camada2, ..., camadaN] de pesos das camadas
                                            - Formato: weights[L][y][x] representa o peso da conexão
                                              do neurônio `x` da camada `L` para o neurônio `y` da camada `L+1`

        bias (List[List[float]]): Lista contendo o bias de cada camada [bias_camada1, bias_camada2, ..., bias_camadaN]
                                            - Formato: bias[L][y] representa o bias da conexão
                                              para o neurônio `y` da camada `L+1`


        neuron_values (List[List[float]]): Lista dos valores de cada neurônio de uma instância
                                            - Lista contendo camadas [neuronio1, neuronio2, ..., neuronioN] de valores
                                            - A camada 0 é referente a camada de entrada
                                            - A camada N é referente a camada de saída (ou última camada intermediária)

        original_bounds (List[List[(float, float)]]): uma lista de lista de tuplas
                                            - [[(x1_min, x1_max) ...][(y1_1_min, y1_1_max)...]...]
                                            - Lista: contém várias listas em que cada lista é uma camada
                                            - e cada item dessa lista é um neurônio em tupla
                                            - e a tupla é o (min, max) dos bounds do neurônio

    Returns:
        (List[int]): uma lista com os inputs da rede que são importantes
    """

    explained_bounds = []

    explained_bounds.append(explain_output_layer(
        output_predicted_index,
        weights[-1],
        bias[-1],
        neuron_values[-2],
        original_bounds[-2:]
    ))
    for k in range(len(weights) - 2, 0, -1):
        explained_bounds.insert(0, explain_intermediate_layer(
            weights[k],
            bias[k],
            neuron_values[k],
            original_bounds[k:k+2],
            explained_bounds[0]
        ))

    bounds, inputs = explain_input_layer(
        weights[0],
        bias[0],
        neuron_values[0],
        original_bounds[:2],
        explained_bounds[0]
    )
    explained_bounds.insert(0, bounds)
    return explained_bounds, inputs

def explain_output_layer(
        output_predicted_index,
        last_weights,
        last_bias,
        previus_layer_values,
        last_2_original_bounds,
        **kwargs
):
    """

    Args:
        output_predicted_index (int): output index da Rede Neural

        last_weights (List[List[float]]): Lista dos pesos para a camada de saída
                                          - Formato: weights[j][i] representa o peso entre
                                            o neurônio `i` da camada `N-1` para o neurônio `j` da camada `N`
                                            Sendo N a camada de saída

        last_bias (List[float]): Valor do bias aplicado a cada neurônio da camada de saída

        previus_layer_values (List[float]): Lista de valores na instância para a camada intermediária final

        last_2_original_bounds (List[List[(float, float)]]): os bounds para as duas últimas camadas
                                            - Formato: last_2_original_bounds[x][y] é uma tupla (lb, ub)
                                              contendo os bounds do neurônio `y` na camada `N-1+x`

        kwargs:
    Returns:
        (List[(float, float)]): lista dos bounds calculados dos neurônios para a instância em questão
                                que garantem a classe predita
    """

    milp = mp.Model()
    if kwargs:
        if "tolerance" in kwargs:
            milp.parameters.simplex.tolerances.feasibility.set(kwargs["tolerance"])

    input_vars, input_constraints, output_vars = insert_constraints(
        milp,
        previus_layer_values,
        last_weights,
        last_bias,
        np.maximum(last_2_original_bounds[0], 0).tolist(),
        last_2_original_bounds[1]
    )

    bounds = []
    # começo da explicação da última camada intermediária
    for j in range(len(previus_layer_values)):
        milp.remove_constraint(input_constraints[j])
        if input_vars[j].ub <= 0:
            bounds.append(last_2_original_bounds[0][j])
            milp.add_constraint(input_constraints[j])
            continue

        for o in range(len(output_vars)):
            if o == output_predicted_index:
                continue

            output_constraint = milp.add_constraint(output_vars[o] >= output_vars[output_predicted_index],
                                                    ctname=f'O_i>=O_j'
                                                    )

            find_interval(milp, input_value=previus_layer_values[j], input_var=input_vars[j])

            milp.remove_constraint(output_constraint)

        # if input_vars[j].lb >= input_values[j] * 0.9999 > 0 or input_vars[j].ub <= input_values[j] * 1.0001:
        #     milp.add_constraint(input_constraints[j])

            # AS linhas comentadas são uma tentativa desesperada de explicar a camada inicial
            # bounds.append((input_values[j], input_values[j]))
            # continue

        if input_vars[j].lb == 0:
            bounds.append((last_2_original_bounds[0][j][0], input_vars[j].ub))
            continue

        bounds.append((input_vars[j].lb, input_vars[j].ub))

    return bounds


def explain_input_layer(
        first_weights,
        first_bias,
        input_layer_values,
        first_2_original_bounds,
        next_layer_bounds,
        **kwargs
):
    """

    Args:
        first_weights (List[List[float]]): Lista dos pesos para a camada intermediária
                                          - Formato: weights[j][i] representa o peso entre
                                            o neurônio `i` da camada `0` para o neurônio `j` da camada `1`

        first_bias (List[float]): Valor do bias aplicado a cada neurônio da camada intermediária

        input_layer_values (List[float]): Lista de valores na instância para a camada de entrada

        first_2_original_bounds (List[List[(float, float)]]): os bounds para as duas primeiras camadas
                                            - Formato: first_2_original_bounds[x][y] é uma tupla (lb, ub)
                                              contendo os bounds do neurônio `y` na camada x

        next_layer_bounds (List[(float, float)]): os bounds calculados para a camada intermediária

    Returns:
        List[(float, float)]: Os bounds encontrados
    """
    milp = mp.Model()
    if kwargs:
        if "tolerance" in kwargs:
            milp.parameters.simplex.tolerances.feasibility.set(kwargs["tolerance"])

    input_vars, input_constraints, output_vars = insert_constraints(
        milp,
        input_layer_values,
        first_weights,
        first_bias,
        first_2_original_bounds[0],
        first_2_original_bounds[1]
    )
    bounds = []
    for j in range(len(input_vars)):
        milp.remove_constraint(input_constraints[j])

        for o in range(len(next_layer_bounds)):

            # y > ub
            lb, ub = next_layer_bounds[o]
            ub = (output_vars[o].ub + ub) / 2
            lb = (output_vars[o].lb + lb) / 2
            if output_vars[o].ub != ub:
                y_ge_ub_constraint = milp.add_constraint(output_vars[o] >= ub, ctname="y_j >= ub'_j")
                to_min_constraint = milp.add_constraint(input_vars[j] >= input_layer_values[j])

                milp.minimize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_min_constraint)
                milp.remove_objective()

                if milp.solution is not None:
                    if milp.solution.get_objective_value() != input_vars[j].ub:
                        milp.add_constraint(input_constraints[j])

                    milp.remove_constraint(y_ge_ub_constraint)
                    break

                to_max_constraint = milp.add_constraint(input_vars[j] <= input_layer_values[j])

                milp.maximize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_max_constraint)
                milp.remove_objective()

                milp.remove_constraint(y_ge_ub_constraint)

                if milp.solution is not None:
                    if milp.solution.get_objective_value() != input_vars[j].lb:
                        milp.add_constraint(input_constraints[j])

                    break

            if output_vars[o].lb != lb:
                # y < lb
                y_le_lb_constraint = milp.add_constraint(output_vars[o] <= lb, ctname="y_j <= lb'_j")
                to_max_constraint = milp.add_constraint(input_vars[j] <= input_layer_values[j])

                milp.maximize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_max_constraint)
                milp.remove_objective()

                if milp.solution is not None:
                    if milp.solution.get_objective_value() != input_vars[j].lb:
                        milp.add_constraint(input_constraints[j])

                    milp.remove_constraint(y_le_lb_constraint)
                    break

                to_min_constraint = milp.add_constraint(input_vars[j] >= input_layer_values[j])

                milp.minimize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_min_constraint)
                milp.remove_objective()

                milp.remove_constraint(y_le_lb_constraint)

                if milp.solution is not None:
                    if milp.solution.get_objective_value() != input_vars[j].ub:
                        milp.add_constraint(input_constraints[j])

                    break

        bounds.append((input_vars[j].lb,input_vars[j].ub))

    return bounds, milp.find_matching_linear_constraints('input')

def explain_intermediate_layer(
        k_weights,
        k_bias,
        input_layer_values,
        original_bounds,
        next_layer_bounds,
        **kwargs
):
    """

    Args:
        k_weights (List[List[float]]): Lista dos pesos para a camada intermediária
                                          - Formato: weights[j][i] representa o peso entre
                                            o neurônio `i` da camada `k` para o neurônio `j` da camada `k+1`

        k_bias (List[float]): Valor do bias aplicado a cada neurônio da camada intermediária

        input_layer_values (List[float]): Lista de valores na instância para a camada de intermediária

        original_bounds (List[List[(float, float)]]): os bounds para as duas primeiras camadas
                                            - Formato: original_bounds[x][y] é uma tupla (lb, ub)
                                              contendo os bounds do neurônio `y` na camada 'k+x'

        next_layer_bounds (List[(float, float)]): os bounds calculados para a camada intermediária 'k+1'

    Returns:
        List[(float, float)]: Os bounds encontrados
    """
    milp = mp.Model()
    if kwargs:
        if "tolerance" in kwargs:
            milp.parameters.simplex.tolerances.feasibility.set(kwargs["tolerance"])

    input_vars, input_constraints, output_vars = insert_constraints(
        milp,
        input_layer_values,
        k_weights,
        k_bias,
        original_bounds[0],
        original_bounds[1]
    )
    bounds = []
    for j in range(len(input_vars)):
        milp.remove_constraint(input_constraints[j])

        for o in range(len(next_layer_bounds)):

            # y > ub
            lb, ub = next_layer_bounds[o]
            ub = (output_vars[o].ub + ub) / 2
            lb = (output_vars[o].lb + lb) / 2
            if output_vars[o].ub != ub:
                y_ge_ub_constraint = milp.add_constraint(output_vars[o] >= ub, ctname="y_j >= ub'_j")
                to_min_constraint = milp.add_constraint(input_vars[j] >= input_layer_values[j])

                milp.minimize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_min_constraint)
                milp.remove_objective()

                if milp.solution is not None:
                    minimized_x_j = milp.solution.get_objective_value()
                    ub_j_minus_e = minimized_x_j - (minimized_x_j - input_layer_values[j]) * 0.5
                    input_vars[j].ub = ub_j_minus_e
                    # input_vars[j].ub = input_layer_values[j]
                    # milp.add_constraint(input_constraints[j])
                    # break

                to_max_constraint = milp.add_constraint(input_vars[j] <= input_layer_values[j])

                milp.maximize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_max_constraint)
                milp.remove_objective()

                milp.remove_constraint(y_ge_ub_constraint)

                if milp.solution is not None:
                    maximized_x_j = milp.solution.get_objective_value()
                    lb_j_plus_e = maximized_x_j + (input_layer_values[j] - maximized_x_j) * 0.5
                    input_vars[j].lb = lb_j_plus_e
                    # input_vars[j].lb = input_layer_values[j]

                    # milp.add_constraint(input_constraints[j])
                    # break

            if output_vars[o].lb != lb:
                # y < lb
                y_le_lb_constraint = milp.add_constraint(output_vars[o] <= lb, ctname="y_j <= lb'_j")
                to_max_constraint = milp.add_constraint(input_vars[j] <= input_layer_values[j])

                milp.maximize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_max_constraint)
                milp.remove_objective()

                if milp.solution is not None:
                    maximized_x_j = milp.solution.get_objective_value()
                    lb_j_plus_e = maximized_x_j + (input_layer_values[j] - maximized_x_j) * 0.5
                    input_vars[j].lb = lb_j_plus_e
                    # input_vars[j].lb = input_layer_values[j]
                    # milp.add_constraint(input_constraints[j])
                    # break

                to_min_constraint = milp.add_constraint(input_vars[j] >= input_layer_values[j])

                milp.minimize(input_vars[j])
                milp.solve()
                milp.remove_constraint(to_min_constraint)
                milp.remove_objective()

                milp.remove_constraint(y_le_lb_constraint)

                if milp.solution is not None:
                    minimized_x_j = milp.solution.get_objective_value()
                    ub_j_minus_e = minimized_x_j - (minimized_x_j - input_layer_values[j]) * 0.5
                    input_vars[j].ub = ub_j_minus_e
                    # input_vars[j].ub = input_layer_values[j]
                    # milp.add_constraint(input_constraints[j])
                    # break

        bounds.append((input_vars[j].lb,input_vars[j].ub))
    return bounds


def insert_constraints(milp, input_values, weights, bias, bounds_in, bounds_out):
    input_vars = []
    input_constraints = []
    output_vars = []
    for i_neuron, (lb, ub) in enumerate(bounds_in):
        input_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'in_{i_neuron}'))
        input_constraints.append(milp.add_constraint(input_vars[i_neuron] == input_values[i_neuron],
                                                     ctname=f'input_{i_neuron}'
                                                     ))

    for j_neuron, (lb, ub) in enumerate(bounds_out):
        output_vars.append(milp.continuous_var(lb=lb, ub=ub, name=f'out_{j_neuron}'))
        milp.add_constraint(output_vars[j_neuron] == weights[j_neuron, :] @ input_vars + bias[j_neuron],
                                ctname=f'weighted_sum_{j_neuron}'
                                )

    return input_vars, input_constraints, output_vars

def find_interval(milp, input_value, input_var, is_relu_layer=True, epsilon=0.5):
        to_min_constraint = milp.add_constraint(input_var >= input_value)

        milp.minimize(input_var)
        milp.solve()
        milp.remove_objective()

        if milp.solution is not None:
            minimized_x_j = milp.solution.get_objective_value()
            ub_j_minus_e = minimized_x_j - (minimized_x_j - input_value) * epsilon
            input_var.ub = ub_j_minus_e

        milp.remove_constraint(to_min_constraint)

        if input_value == 0 and is_relu_layer:
            return

        milp.maximize(input_var)
        milp.solve()
        milp.remove_objective()
        if milp.solution is not None:
            maximized_x_j = milp.solution.get_objective_value()
            lb_j_plus_e = maximized_x_j + (input_value - maximized_x_j) * epsilon
            input_var.lb = lb_j_plus_e
