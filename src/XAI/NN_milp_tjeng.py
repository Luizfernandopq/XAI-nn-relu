from src.XAI.NN_milp import NN_milp


class NN_milp_tjeng(NN_milp):
    def __init__(self, network, dataframe):
        super().__init__(network, dataframe)
        self.output_bounds = []

    def codify_milp_network(self):
        super().codify_milp_network()
        len_layers = len(self.network.layers)
        for i in range(len_layers):
            A = self.network.layers[i].get_weights()[0].T
            b = self.network.layers[i].bias.numpy()

            x = self.input_variables if i == 0 else self.intermediate_variables[i - 1]

            if i != (len_layers - 1):
                a = self.decision_variables[i]
                y = self.intermediate_variables[i]
            else:
                y = self.output_variables

            for j in range(A.shape[0]):

                weighted_sum = A[j, :] @ x + b[j]
                ub = self.maximize(weighted_sum)

                if ub <= 0 and i != len_layers - 1:
                    self.milp_repr.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')
                    continue

                lb = self.minimize(weighted_sum)

                if lb >= 0 and i != len_layers - 1:
                    self.milp_repr.add_constraint(weighted_sum == y[j], ctname=f'c_{i}_{j}')
                    continue

                if i != len_layers - 1:
                    self.milp_repr.add_constraint(y[j] <= weighted_sum - lb * (1 - a[j]))
                    self.milp_repr.add_constraint(y[j] >= weighted_sum)
                    self.milp_repr.add_constraint(y[j] <= ub * a[j])
                else:
                    self.milp_repr.add_constraint(weighted_sum == y[j])
                    self.output_bounds.append([lb, ub])

    def maximize(self, expr):
        self.milp_repr.maximize(expr)
        self.milp_repr.solve()
        ub = self.milp_repr.solution.get_objective_value()
        self.milp_repr.remove_objective()
        return ub

    def minimize(self, expr):
        self.milp_repr.minimize(expr)
        self.milp_repr.solve()
        lb = self.milp_repr.solution.get_objective_value()
        self.milp_repr.remove_objective()
        return lb

    def insert_output_constraints(self, index_output_predicted, binary_variables):
        variable_output_predicted = self.output_variables[index_output_predicted]

        # Output i: oi - oj <= u1 = ui - lj
        upper_bounds_diffs = self.output_bounds[index_output_predicted][1] - np.array(self.output_bounds)[:, 0]
        aux_var = 0

        for i, output in enumerate(self.output_variables):
            if i != index_output_predicted:
                ub = upper_bounds_diffs[i]
                z = binary_variables[aux_var]
                self.milp_repr.add_constraint(variable_output_predicted - output - ub * (1 - z) <= 0)
                aux_var += 1

        return self.milp_repr
