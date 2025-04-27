import numpy as np

from decision_boundary_plotter import DecisionBoundaryPlotter
from perceptron import Perceptron

training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
expected_outputs = np.array([0, 1, 1, 1])

def main():
    perceptron = Perceptron(num_inputs=2)

    print("Entrenando para predecir compuerta OR")
    #Train
    perceptron.train(training_inputs, expected_outputs, epochs=10)

    #Plotter
    plotter = DecisionBoundaryPlotter(perceptron, training_inputs, expected_outputs)

    #Predict
    print("Pruebas:")
    for inputs in training_inputs:
        prediction = perceptron.predict(inputs)
        print(f"Entrada: {inputs}, Predicción: {prediction}")

    #Plot decision boundary
    plotter.plot()


if __name__ == "__main__":
    main()
