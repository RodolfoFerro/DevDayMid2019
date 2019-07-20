import numpy as np


class SigmoidNeuron():
    def __init__(self, n):
        np.random.seed(123)
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_output, iterations):
        for iteration in range(iterations):
            output = self.predict(training_inputs)
            error = training_output.reshape((len(training_inputs), 1)) - output
            adjustment = np.dot(training_inputs.T, error *
                                self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    # Initialize Sigmoid Neuron:
    sigmoid = SigmoidNeuron(2)
    print("Inicialización de pesos aleatorios:")
    print(sigmoid.synaptic_weights)

    # Datos de entrenamiento:
    training_inputs = np.array([[1, 0], [0, 0], [0, 1]])
    training_output = np.array([1, 0, 1]).T.reshape((3, 1))

    # Entrenamos la neurona (100,000 iteraciones):
    sigmoid.train(training_inputs, training_output, 100000)
    print("Nuevos pesos sinápticos luego del entrenamiento: ")
    print(sigmoid.synaptic_weights)

    # Predecimos para probar la red:
    print("Predicción para [1, 1]: ")
    print(sigmoid.predict(np.array([1, 1])))
