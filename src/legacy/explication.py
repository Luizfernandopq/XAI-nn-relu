import numpy as np


def insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds):
    variable_output = output_variables[network_output]
    upper_bounds_diffs = output_bounds[network_output][1] - np.array(output_bounds)[:,
                                                            0]  # Output i: oi - oj <= u1 = ui - lj
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            z = binary_variables[aux_var]
            mdl.add_constraint(variable_output - output - ub * (1 - z) <= 0)
            aux_var += 1

    return mdl

def get_miminal_explanation(mdl, network_input, network_output, output_bounds, n_classes):

    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input))]
    input_constraints = mdl.add_constraints([
        input_variables[i] == feature for i, feature in enumerate(network_input)], names='input')
    binary_variables = mdl.binary_var_list(3 - 1, name='b')

    mdl.add_constraint(mdl.sum(binary_variables) >= 1)
    mdl = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables,
                                              output_bounds)

    for i in range(len(network_input)):
        mdl.remove_constraint(input_constraints[i])

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(input_constraints[i])

    return mdl.find_matching_linear_constraints('input')
