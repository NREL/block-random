import pickle


# ===============================================================================
#
# Function definitions
#
# ===============================================================================
def load_history(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


# ===============================================================================
def load_parameters(fname):
    parameters = {
        "epochs": 0,
        "batch_size": 0,
        "learning_rate": 0,
        "n_train": 0,
        "n_test": 0,
        "n_layers": 0,
        "n_nodes": 0,
        "rebalance": None,
        "permutation": None,
        "datadir": None,
        "logdir": None,
    }
    with open(fname, "r") as f:
        for line in f:
            for param in parameters:
                if param in line:
                    if param in [
                        "epochs",
                        "batch_size",
                        "n_train",
                        "n_test",
                        "n_layers",
                        "n_nodes",
                    ]:
                        parameters[param] = int(line.split()[-1])
                    elif param in ["learning_rate"]:
                        parameters[param] = float(line.split()[-1])
                    elif param in ["permutation", "rebalance", "datadir", "logdir"]:
                        parameters[param] = line.split()[-1]

    return parameters
