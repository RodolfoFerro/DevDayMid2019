import numpy as np


class Perceptron():
    def __init__(self, entradas, pesos):
        """Constructor de la clase."""
        self.n = len(entradas)
        self.entradas = np.array(entradas)
        self.pesos = np.array(pesos)

    def simon(self, umbral):
        """Simón dice."""
        return (self.entradas @ self.pesos) >= umbral


if __name__ == '__main__':
    """
    a) Lluvia
    b) Amigos
    c) Cheves
    d) Pláticas cool
    """
    entradas = [1, 1, 0, 1]
    pesos = [-2, 3, -11, 3]
    print(entradas)
    print(pesos)

    p = Perceptron(entradas, pesos)
    print(p.simon(5))
