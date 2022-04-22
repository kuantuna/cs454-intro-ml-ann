from layer import Layer
import numpy as np


def main() -> None:
    network = [Layer(1, 3), Layer(3, 2), Layer(2, 1)]
    inp = np.array([[2]])
    for layer in network:
        inp = layer.forward(inp)
        print(inp)


if __name__ == "__main__":
    main()
