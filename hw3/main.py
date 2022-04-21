import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from dense import Dense


def main() -> None:
    """Reading the training data"""
    train_dfs: list[pd.DataFrame] = []
    for i in range(1, 11):
        train_dfs.append(pd.read_csv(f"sample{i}.csv", header=None))

    """Reading the test data"""
    test_df: pd.DataFrame = pd.read_csv("test.csv", header=None)

    """Dividing the training data into X and y"""
    train_dfs_xs: list[np.ndarray] = []
    train_dfs_ys: list[np.ndarray] = []
    for train_df in train_dfs:
        train_dfs_xs.append(train_df.iloc[:, :-1].to_numpy())
        train_dfs_ys.append(train_df.iloc[:, -1].to_numpy())

    """Dividing the test data into X and y"""
    test_df_xs: np.ndarray = test_df.iloc[:, :-1].to_numpy()
    test_df_ys: np.ndarray = test_df.iloc[:, -1].to_numpy()

    """Finishing the data preperation"""
    train_X: np.ndarray = np.array(train_dfs_xs)
    train_y: np.ndarray = np.array(train_dfs_ys).reshape(10, 25, 1)

    """Creating the network for SingleLayerPerceptron"""
    slp_network = [
        Dense(input_size=1, output_size=2),
        # Activation,
        Dense(input_size=2, output_size=1),
        # Activation
    ]

    """Creating the network for MultiLayerPerceptrons"""
    mlp_network_2 = [
        Dense(input_size=1, output_size=2),
        # Activation,
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=1),
        # Activation
    ]

    mlp_network_3 = [
        Dense(input_size=1, output_size=2),
        # Activation,
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=1),
        # Activation
    ]

    mlp_network_5 = [
        Dense(input_size=1, output_size=2),
        # Activation,
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=2),
        # Activation
        Dense(input_size=2, output_size=1),
        # Activation
    ]


if __name__ == '__main__':
    main()
