# Task 1

## Running the app
The following task required us to make a streamlit app to visualize the workings of neural networks and its dependence on various parameters.

The streamlit app can be run locally by running the following command in the terminal:

```sh
streamlit run streamlit_app.py
```

## App Description

The app has 3 main sections:

### 1. Dataset Visualization

When opening the app for the first time or whenever its parameters are changed/reset, the app will display the dataset visualization section. This section contains a scatter plot of the dataset that has been selected.

### 2. Adjust Hyperparameters

The sidebar on the streamlit app contains different parameters that can be adjusted to change the neural network. The parameters are:
- Number of hidden layers
- Number of neurons in each hidden layer (For each hidden layer, the number of neurons can be adjusted)
- Basis function (this contains the different features that can be used as the input to the neural network. This explore the realm of feature expansion)
    - x1, x2
    - x1^2, x2^2
    - x1x2
    - sin(x1), sin(x2)
    - gaussian(x1), gaussian(x2) (This is a gaussian basis function with mean 0 and variance 1, i.e. N(0,1), gaussian(x) = exp(-x^2/2))

- Dataset (The different datasets that can be used to train the neural network)
    - Gaussian
    - Moons
    - Circles
    - Spiral

- Learning rate (The learning rate of the neural network)
- Number of epochs (The number of epochs for which the neural network will be trained)
- Input Noise
- Number of samples in the input dataset
- MC Dropout Layer (Checking this will add a dropout layer after each hidden layer. This is used to check the effect of dropout on the neural network. Monte Carlo Dropout is used for inference too)

### 3. Get the Classification Boundary

Click the button to get the classification boundary. This will train the neural network with the given parameters and display the classification boundary with the heatmap representing the probabilities of the classes.