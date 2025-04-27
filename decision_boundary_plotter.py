import matplotlib.pyplot as plt
import numpy as np

class DecisionBoundaryPlotter:
    def __init__(self, perceptron, training_inputs, expected_outputs):
        self.perceptron = perceptron
        self.training_inputs = training_inputs
        self.expected_outputs = expected_outputs

    def plot(self):
        # Points
        for inputs, expected_output in zip(self.training_inputs, self.expected_outputs):
            if expected_output == 0:
                plt.scatter(inputs[0], inputs[1], color='red', marker='o', label='Clase 0' if 'Clase 0' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(inputs[0], inputs[1], color='blue', marker='x', label='Clase 1' if 'Clase 1' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Decision boundary
        x_values = np.linspace(-0.5, 1.5, 100)

        if self.perceptron.weights[1] != 0:
            y_values = -(self.perceptron.weights[0]/self.perceptron.weights[1]) * x_values - (self.perceptron.bias/self.perceptron.weights[1])
            plt.plot(x_values, y_values, color='green', label='Frontera de decisión')
        else:
            x_constant = -self.perceptron.bias / self.perceptron.weights[0]
            plt.axvline(x=x_constant, color='green', label='Frontera de decisión')

        # Accuracy
        accuracy = self.perceptron.calculate_accuracy(self.training_inputs, self.expected_outputs)

        # Graph settings
        plt.xlabel('Entrada 1')
        plt.ylabel('Entrada 2')
        plt.title(f'Frontera de Decisión - Exactitud: {accuracy * 100:.2f}%')
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.legend()
        plt.grid(True)
        plt.show()