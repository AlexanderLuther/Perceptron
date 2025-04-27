import numpy as np
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

    #Predict
    print("Pruebas:")
    for inputs in training_inputs:
        prediction = perceptron.predict(inputs)
        print(f"Entrada: {inputs}, Predicción: {prediction}")


if __name__ == "__main__":
    main()
