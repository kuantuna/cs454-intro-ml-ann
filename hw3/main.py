import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer import Layer, SigmoidActivationLayer, mse, mse_derivative


def train(epochs, learning_rate, train_df_xs, train_df_ys, network):
    for _ in range(epochs):
        for x, y in zip(train_df_xs, train_df_ys):
            output = x
            for layer in network:
                output = layer.forward(output)

            grad = mse_derivative(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)


def test(test_df_xs, test_df_ys, network, network_name, outputs, df_i):
    predictions = []
    error = 0
    for x, y in zip(test_df_xs, test_df_ys):
        output = x
        for layer in network:
            output = layer.forward(output)

        predictions.append(output.item())
        error += mse(y, output)
    error /= len(test_df_xs)
    outputs[network_name][df_i]["x"] = test_df_xs.reshape(
        test_df_xs.shape[0]).tolist()
    outputs[network_name][df_i]["y"].extend(predictions)
    outputs[network_name][df_i]["error"] = error.item()
    print(f"TEST ERROR={error}")


def addLabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i], 2), ha='center')


def main() -> None:
    """Reading the training data"""
    train_dfs: list[pd.DataFrame] = []
    for i in range(1, 11):
        train_dfs.append(pd.read_csv(f"data/sample{i}.csv", header=None))

    """Reading the test data"""
    test_df: pd.DataFrame = pd.read_csv("data/test.csv", header=None)

    """Dividing the training data into X and y"""
    train_dfs_xs: list[np.ndarray] = []
    train_dfs_ys: list[np.ndarray] = []
    for train_df in train_dfs:
        trdx = train_df.iloc[:, :-1].to_numpy()  # shape = 25, 1
        trdy = train_df.iloc[:, -1].to_numpy()   # shape = 25,
        trdy = trdy.reshape(trdy.shape[0], 1)               # shape = 25, 1
        train_dfs_xs.append(trdx.reshape(trdx.shape[0], trdx.shape[1], 1))
        train_dfs_ys.append(trdy.reshape(trdy.shape[0], trdy.shape[1], 1))

    """Dividing the test data into X and y"""
    test_df_xs: np.ndarray = test_df.iloc[:, :-1].to_numpy()
    test_df_xs = test_df_xs.reshape(
        test_df_xs.shape[0], test_df_xs.shape[1], 1)
    test_df_ys: np.ndarray = test_df.iloc[:, -1].to_numpy()
    test_df_ys = test_df_ys.reshape(test_df_ys.shape[0], 1, 1)

    slp = [Layer(1, 1)]
    mlp_2 = [Layer(1, 2), SigmoidActivationLayer(), Layer(2, 1)]
    mlp_3 = [Layer(1, 3), SigmoidActivationLayer(), Layer(3, 1)]
    mlp_5 = [Layer(1, 5), SigmoidActivationLayer(), Layer(5, 1)]
    mlp_10 = [Layer(1, 10), SigmoidActivationLayer(), Layer(10, 1)]

    network_names = ["slp", "mlp_2", "mlp_3", "mlp_5", "mlp_10"]
    networks = [slp, mlp_2, mlp_3, mlp_5, mlp_10]

    epochs = 10
    learning_rate = 0.001
    outputs = {name: {i+1: {"x": [], "y": [], "error": 0.0}
                      for i in range(10)} for name in network_names}

    for df_i, (train_df_xs, train_df_ys) in enumerate(zip(train_dfs_xs, train_dfs_ys)):
        # For every train df of 25x1
        print(f"\nDf{df_i + 1}")

        for i, (network, network_name) in enumerate(zip(networks, network_names)):
            # For every network
            print(f"\nNetwork{i + 1}")

            # Training
            train(epochs, learning_rate, train_df_xs, train_df_ys, network)

            # Testing
            test(test_df_xs, test_df_ys, network,
                 network_name, outputs, df_i + 1)

    mse_values_by_network = {network_name: []
                             for network_name in network_names}

    for name, df_indexes in outputs.items():
        for df_index, df_data in df_indexes.items():
            mse_values_by_network[name].append(df_data["error"])
            plt.title(
                f"Plot of g(x)s on 10 different training dfs for = {name}", fontsize=20)
            plt.plot(df_data["x"], df_data["y"], label=f"Train Df: {df_index}")
            plt.legend(loc='upper left')
        plt.show()
    mean_squared_errors = [np.mean(mses)
                           for mses in mse_values_by_network.values()]
    plt.title(f"Plot of Average MSE values for different sigma values", fontsize=20)
    plt.bar(range(5), mean_squared_errors)
    plt.xticks(range(5), [0, 2, 3, 5, 10])
    plt.xlabel('Sigma (h) value', fontsize=15)
    plt.ylabel('Average MSE', fontsize=15)
    addLabels([0, 2, 3, 5, 10], mean_squared_errors)
    plt.show()


if __name__ == "__main__":
    main()
