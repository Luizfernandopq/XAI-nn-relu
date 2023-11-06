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
        pass
