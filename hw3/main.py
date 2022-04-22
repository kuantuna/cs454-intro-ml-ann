from layer import Layer
import numpy as np
import pandas as pd


def main() -> None:
    """Reading the training data"""
    train_dfs: list[pd.DataFrame] = []
    for i in range(1, 11):
        train_dfs.append(pd.read_csv(f"../data/sample{i}.csv", header=None))

    """Reading the test data"""
    test_df: pd.DataFrame = pd.read_csv("../data/test.csv", header=None)

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

    # for y in train_dfs_ys:
    #     print(y)
    #     print(f"{y.shape}\n")

    slp = [Layer(1, 1, is_output=True)]
    mlp_2 = [Layer(1, 2), Layer(2, 1, is_output=True)]
    mlp_3 = [Layer(1, 3), Layer(3, 1, is_output=True)]
    mlp_5 = [Layer(1, 5), Layer(5, 1, is_output=True)]
    mlp_10 = [Layer(1, 10), Layer(10, 1, is_output=True)]

    networks = [slp, mlp_2, mlp_3, mlp_5, mlp_10]
    # network = [Layer(1, 3), Layer(3, 2), Layer(2, 1, is_output=True)]
    for network in networks:
        for x in train_dfs_xs:
            inp = x
            for layer in network:
                inp = layer.forward(inp)
                print(inp)
            print("\n")
            break


if __name__ == "__main__":
    main()
