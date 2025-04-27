# Perceptrón para Compuerta OR

Este proyecto implementa un perceptrón simple en Python para aprender y predecir la función lógica OR.

## Descripción

El proyecto consta de dos archivos principales:

*   `main.py`: Contiene el código principal para entrenar y probar el perceptrón.
*   `perceptron.py`: Define la clase `Perceptron` que implementa la lógica del perceptrón.

El perceptrón se entrena utilizando un conjunto de datos de entrenamiento que representa las entradas y salidas esperadas para la compuerta OR. Después del entrenamiento, el perceptrón puede predecir la salida para nuevas entradas.

## Uso

1.  Tener Python y NumPy instalados.
2.  Guarda los archivos `main.py` y `perceptron.py` en el mismo directorio.
3.  Ejecuta el archivo `main.py`:

    ```bash
    python main.py
    ```

## Estructura del Código

### `perceptron.py`

La clase `Perceptron` tiene los siguientes métodos:

*   `__init__(self, num_inputs, learning_rate=0.1)`: Inicializa el perceptrón con un número dado de entradas y una tasa de aprendizaje opcional. Los pesos se inicializan en cero y el bias también se inicializa en cero.
*   `predict(self, inputs)`: Realiza una predicción basada en las entradas dadas. Calcula la suma ponderada de las entradas más el bias y aplica la función escalón.
*   `train(self, training_inputs, expected_outputs, epochs)`: Entrena el perceptrón utilizando los datos de entrenamiento proporcionados durante un número específico de épocas. Ajusta los pesos y el bias basándose en el error entre la predicción y la salida esperada.
*   `__step_function(self, x)`: Implementa la función escalón, que devuelve 1 si la entrada es mayor o igual a 0, y 0 en caso contrario.

### `main.py`

El archivo `main.py` define los datos de entrenamiento y utiliza la clase `Perceptron` para entrenar y probar el perceptrón.

*   Se define un conjunto de datos de entrenamiento `training_inputs` que representa las posibles entradas para la compuerta OR.
*   Se definen las salidas esperadas `expected_outputs` correspondientes a las entradas de entrenamiento.
*   Se crea una instancia de la clase `Perceptron` con dos entradas.
*   Se entrena el perceptrón utilizando los datos de entrenamiento y las salidas esperadas durante 10 épocas.
*   Se realizan predicciones para cada entrada en el conjunto de datos de entrenamiento y se imprimen los resultados.

## Dependencias

*   NumPy