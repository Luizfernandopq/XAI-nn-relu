import random

import numpy as np
import docplex.mp.model as mp
from cplex import infinity


def codify_network_tjeng(mdl, layers, input_variables, intermediate_variables, decision_variables, output_variables):
    output_bounds = []

    for i in range(len(layers)):
        A = layers[i].weight.detach().numpy()
        b = layers[i].bias.detach().numpy()

        x = input_variables if i == 0 else intermediate_variables[i-1]

        if i != len(layers) - 1:
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):

            mdl.maximize(A[j, :] @ x + b[j])
            mdl.solve()
            ub = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if ub <= 0 and i != len(layers) - 1:
                 mdl.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')
                 continue

            mdl.minimize(A[j, :] @ x + b[j])
            mdl.solve()
            lb = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if lb >= 0 and i != len(layers) - 1:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
                continue

            if i != len(layers) - 1:
                mdl.add_constraint(y[j] <= A[j, :] @ x + b[j] - lb * (1 - a[j]))
                mdl.add_constraint(y[j] >= A[j, :] @ x + b[j])
                mdl.add_constraint(y[j] <= ub * a[j])
            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j])

                output_bounds.append([lb, ub])

    return mdl, output_bounds


def relaxed_codify_network(network, dataframe, relax_density=0.25):
    layers = network.layers
    mdl = mp.Model()

    _, bounds_input = get_types_and_bounds(dataframe)
    bounds_input = np.array(bounds_input)

    input_variables = []
    # Pressuposto de ignorar datasets com variáveis não contínuas
    for index, (lb, ub) in enumerate(bounds_input):
        input_variables.append(mdl.continuous_var(lb=lb, ub=ub, name=f'x_{index}'))

    intermediate_variables = []
    decision_variables = []

    for i in range(len(layers)-1):
        weights = layers[i].weight.detach().numpy().T
        intermediate_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))
        decision_variables.append(mdl.binary_var_list(weights.shape[1], name='a', lb=0, ub=1, key_format=f"_{i}_%s"))

    output_variables = mdl.continuous_var_list(layers[-1].weight.detach().numpy().T.shape[1], lb=-infinity, name='o')

    assert 0 <= relax_density <= 1 , "Densidade da relaxação deve estar no intervalo {0, 1}"
    if relax_density > 0.0:
        for binary_vars in decision_variables:
            num_relaxes = int(len(binary_vars) * relax_density)
            relaxes = random.sample(range(0, len(binary_vars)), num_relaxes)

            for index in relaxes:
                binary_vars[index].set_vartype("Continuous")

    mdl, output_bounds = codify_network_tjeng(mdl, layers, input_variables,
                                              intermediate_variables, decision_variables,
                                              output_variables)
    return mdl, output_bounds

def get_types_and_bounds(dataframe, ignore_int=True):
        input_types = []
        bounds = []
        for column in dataframe.columns:
            unique_values = dataframe[column].unique()
            input_types.append(
                'B' if len(unique_values) == 2 else
                'C' if np.any(unique_values.astype(np.int64) != unique_values.astype(np.float64)) or ignore_int else
                'I')
            bounds.append((dataframe[column].min(), dataframe[column].max()))
        return input_types, bounds
