import numpy as np
import pandas as pd

from layer import Layer, SigmoidActivationLayer, mse, mse_derivative


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
        train_dfs_xs.extend(trdx.reshape(trdx.shape[0], trdx.shape[1], 1))
        train_dfs_ys.extend(trdy.reshape(trdy.shape[0], trdy.shape[1], 1))

    """Dividing the test data into X and y"""
    test_df_xs: np.ndarray = test_df.iloc[:, :-1].to_numpy()
    test_df_ys: np.ndarray = test_df.iloc[:, -1].to_numpy()

    slp = [Layer(1, 1)]
    mlp_2 = [Layer(1, 2), SigmoidActivationLayer(), Layer(2, 1)]
    mlp_3 = [Layer(1, 3), SigmoidActivationLayer(), Layer(3, 1)]
    mlp_5 = [Layer(1, 5), SigmoidActivationLayer(), Layer(5, 1)]
    mlp_10 = [Layer(1, 10), SigmoidActivationLayer(), Layer(10, 1)]

    networks = [slp, mlp_2, mlp_3, mlp_5, mlp_10]

    epochs = 1000
    learning_rate = 0.001
    for i, network in enumerate(networks):
        print(f"\n\nNetwork {i + 1}")
        for e in range(epochs):
            error = 0
            for x, y in zip(train_dfs_xs, train_dfs_ys):
                output = x
                for layer in network:
                    output = layer.forward(output)

                error += mse(y, output)

                grad = mse_derivative(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(train_dfs_xs)
            print(f"{e + 1}/{epochs}, error={error}")


if __name__ == "__main__":
    main()
