from src.XAI.NN_milp import NN_milp


class NN_milp_tjeng(NN_milp):
    def __init__(self, network, dataframe):
        super().__init__(network, dataframe)
        self.output_bounds = []

    def codify_milp_network(self):
        super().codify_milp_network()
        for i in range(len(self.network.layers)):
            A = self.network.layers[i].get_weights()[0].T
            b = self.network.layers[i].bias.numpy()

            x = self.input_variables if i == 0 else self.intermediate_variables[i - 1]

            if i != (len(self.network.layers) - 1):
                a = self.decision_variables[i]
                y = self.intermediate_variables[i]
            else:
                y = self.output_variables

            for j in range(A.shape[0]):

                self.milp_repr.maximize(A[j, :] @ x + b[j])
                self.milp_repr.solve()
                ub = self.milp_repr.solution.get_objective_value()
                self.milp_repr.remove_objective()

                if ub <= 0 and i != len(self.network.layers) - 1:
                    self.milp_repr.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')
                    continue

                self.milp_repr.minimize(A[j, :] @ x + b[j])
                self.milp_repr.solve()
                lb = self.milp_repr.solution.get_objective_value()
                self.milp_repr.remove_objective()

                if lb >= 0 and i != len(self.network.layers) - 1:
                    self.milp_repr.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
                    continue

                if i != len(self.network.layers) - 1:
                    self.milp_repr.add_constraint(y[j] <= A[j, :] @ x + b[j] - lb * (1 - a[j]))
                    self.milp_repr.add_constraint(y[j] >= A[j, :] @ x + b[j])
                    self.milp_repr.add_constraint(y[j] <= ub * a[j])
                else:
                    self.milp_repr.add_constraint(A[j, :] @ x + b[j] == y[j])
                    self.output_bounds.append([lb, ub])

    def insert_output_constraints(self, network_output_argmax):
        pass
