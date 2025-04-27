import numpy as np

class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = np.zeros(num_inputs)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.__step_function(summation)

    def train(self, training_inputs, expected_outputs, epochs):
        for epoch in range(epochs):
            for inputs, expected_output in zip(training_inputs, expected_outputs):
                prediction = self.predict(inputs)
                error = expected_output - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    def calculate_accuracy(self, inputs, expected_outputs):
        predictions = [self.predict(x) for x in inputs]
        correct = sum(p == o for p, o in zip(predictions, expected_outputs))
        accuracy = correct / len(expected_outputs)
        return accuracy

    def __step_function(self, x):
        return np.where(x >= 0, 1, 0)