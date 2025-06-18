import time

import numpy as np
import docplex.mp.model as mp
from cplex import infinity


def codify_network_tjeng(mdl, layers, input_variables, intermediate_variables, decision_variables, output_variables):
    output_bounds = []

    for i in range(len(layers)):
        # A = layers[i].get_weights()[0].T
        A = layers[i].weight.detach().numpy()

        # b = layers[i].bias.numpy()
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

            # if ub <= 0 and i != len(layers) - 1:
            #      mdl.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')
            #      continue

            mdl.minimize(A[j, :] @ x + b[j])
            mdl.solve()
            lb = mdl.solution.get_objective_value()
            mdl.remove_objective()

            # if lb >= 0 and i != len(layers) - 1:
            #     mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
            #     continue

            if i != len(layers) - 1:
                if ub <= 0:
                    mdl.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')
                elif lb >= 0:
                    mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
                else:
                    mdl.add_constraint(y[j] <= A[j, :] @ x + b[j] - lb * (1 - a[j]))
                    mdl.add_constraint(y[j] >= A[j, :] @ x + b[j])
                    mdl.add_constraint(y[j] <= ub * a[j])

                #mdl.maximize(y[j])
                #mdl.solve()
                #ub_y = mdl.solution.get_objective_value()
                #mdl.remove_objective()
                #y[j].set_ub(ub_y)

            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j])
                #y[j].set_ub(ub)
                #y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return mdl, output_bounds


def codify_network(model, dataframe):
    layers = model.layers
    num_features = layers[0].weight.detach().numpy().T.shape[0]
    # num_features = layers[0].get_weights()[0].shape[0]
    mdl = mp.Model()

    domain_input, bounds_input = get_domain_and_bounds_inputs(dataframe)
    bounds_input = np.array(bounds_input)

    input_variables = []
    for i in range(len(domain_input)):
        lb, ub = bounds_input[i]
        if domain_input[i] == 'C':
            input_variables.append(mdl.continuous_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 'I':
            input_variables.append(mdl.integer_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 'B':
            input_variables.append(mdl.binary_var(name=f'x_{i}'))
    intermediate_variables = []
    auxiliary_variables = []
    decision_variables = []

    for i in range(len(layers)-1):
        # weights = layers[i].get_weights()[0]
        weights = layers[i].weight.detach().numpy().T
        intermediate_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))

        decision_variables.append(mdl.binary_var_list(weights.shape[1], name='a', lb=0, ub=1, key_format=f"_{i}_%s"))

    # output_variables = mdl.continuous_var_list(layers[-1].get_weights()[0].shape[1], lb=-infinity, name='o')
    output_variables = mdl.continuous_var_list(layers[-1].weight.detach().numpy().T.shape[1], lb=-infinity, name='o')

    mdl, output_bounds = codify_network_tjeng(mdl, layers, input_variables,
                                                  intermediate_variables, decision_variables, output_variables)
    return mdl, output_bounds


def get_domain_and_bounds_inputs(dataframe):
    domain = []
    bounds = []
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 2:
            domain.append('B')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        elif np.any(dataframe[column].unique().astype(np.int64) != dataframe[column].unique().astype(np.float64)):
            domain.append('C')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        else:
            domain.append('I')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])

    return domain, bounds
