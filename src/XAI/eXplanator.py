import NN_milp
from src.XAI.NN_milp_tjeng import NN_milp_tjeng


class eXplanator:
    def __init__(self, network, dataframe, **kwargs):
        kwargs = {"metodo": "tjeng"} if not kwargs else kwargs
        match kwargs["metodo"]:
            case "tjeng":
                self.milp = NN_milp_tjeng(network, dataframe)
                self.milp.codify_milp_network()

    def get_minimal_explanation(self, network_input, network_output):
        len_output = len(self.milp.output_bounds)
        input_variables = [self.milp.milp_repr.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
        input_constraints = self.milp.milp_repr.add_constraints(
            [input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])], names='input')
        binary_variables = self.milp.milp_repr.binary_var_list(len_output - 1, name='b')

        self.milp.milp_repr.add_constraint(self.milp.milp_repr.sum(binary_variables) >= 1)
        self.milp.milp_repr.insert_output_constraints_tjeng(network_output, binary_variables)

        for i in range(len(network_input[0])):
            self.milp.milp_repr.remove_constraint(input_constraints[i])

            self.milp.milp_repr.solve(log_output=False)
            if self.milp.milp_repr.solution is not None:
                self.milp.milp_repr.add_constraint(input_constraints[i])

        return self.milp.milp_repr.find_matching_linear_constraints('input')
