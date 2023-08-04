import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from data_generation import DataGenerator
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def mlp_model(architecture_info, input_dim, learning_rate, mc_dropout):
    model = Sequential()
    model.add(Dense(input_dim, activation='relu'))
    for i in range(len(architecture_info)):
        model.add(Dense(architecture_info[i]['neurons'], activation=architecture_info[i]['activation']))
        #Add Dropout layer
        if mc_dropout:
            model.add(MCDropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def transformations(data_points, functions):
    transformed_data_points = []
    if 'x1' in functions:
        transformed_data_points.append(data_points[:, 0])
    if 'x2' in functions:
        transformed_data_points.append(data_points[:, 1])
    if 'x1^2' in functions:
        transformed_data_points.append(data_points[:, 0] ** 2)
    if 'x2^2' in functions:
        transformed_data_points.append(data_points[:, 1] ** 2)
    if 'x1x2' in functions:
        transformed_data_points.append(data_points[:, 0] * data_points[:, 1])
    if 'sin(x1)' in functions:
        transformed_data_points.append(np.sin(data_points[:, 0]))
    if 'sin(x2)' in functions:
        transformed_data_points.append(np.sin(data_points[:, 1]))
    if 'gaussian(x1)' in functions:
        transformed_data_points.append(np.exp(-data_points[:, 0] ** 2))
    if 'gaussian(x2)' in functions:
        transformed_data_points.append(np.exp(-data_points[:, 1] ** 2))

    return np.array(transformed_data_points).T

# Function to generate the contour plot based on user inputs
def generate_contour_plot(model, data_points, labels, functions, mc_dropout):
    x_min, x_max = data_points[:, 0].min() - 1, data_points[:, 0].max() + 1
    y_min, y_max = data_points[:, 1].min() - 1, data_points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # transform the meshgrid points based on the functions selected
    transformed_grid_points = transformations(np.c_[xx.ravel(), yy.ravel()], functions)

    if mc_dropout:
        total_predictions = []
        for i in range(5):
            total_predictions.append(model.predict(transformed_grid_points))
        print(total_predictions[0].shape)
        predictions = np.mean(total_predictions, axis=0)
    else:
        predictions = model.predict(transformed_grid_points)

    # Reshape the predictions to match the shape of the meshgrid
    zz = predictions.reshape(xx.shape)

    # Plot the contour plot
    fig = plt.figure(figsize=(8, 6))

    # Create the heatmap using plt.imshow
    plt.imshow(zz, cmap='viridis', extent=[xx.min(), xx.max(), yy.min(), yy.max()], origin='lower', aspect='auto', alpha=0.6)

    # Add colorbar for reference
    plt.colorbar()
    plt.scatter(data_points[labels == 1, 0], data_points[labels == 1, 1], c='orange', label='Class 1')
    plt.scatter(data_points[labels == 0, 0], data_points[labels == 0, 1], c='purple', label='Class -1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Heatmap with Classification Prpbabilities')
    plt.legend()
    plt.grid(True)
    # st.pyplot(fig)
    # plt.show()
    return fig

# Main Streamlit app
def main():
    st.title('Neural Networks Playground')

    # Create a sidebar
    st.sidebar.title('Adjust Hyperparameters')

    # Plus and minus buttons for Hidden Layers in the sidebar
    num_hidden_layers = st.sidebar.number_input('Number of Hidden Layers', min_value=1, step=1, value=1)

    # Dynamic creation of hidden layers and their neurons in the sidebar
    architecture_info = []
    for i in range(num_hidden_layers):
        layer_neurons = st.sidebar.number_input(f'Number of neurons in Hidden Layer {i+1}', min_value=1, step=1, value=10)
        architecture_info.append({'neurons': layer_neurons, 'activation': 'relu'})

    # Checkboxes for functions in the sidebar
    functions = st.sidebar.multiselect('Basis Functions', ['x1', 'x2', 'x1^2', 'x2^2', 'x1x2', 'sin(x1)', 'sin(x2)', 'gaussian(x1)', 'gaussian(x2)'], default=['x1', 'x2'])

    dataset = st.sidebar.selectbox('Dataset', ['Gaussian', 'Moons', 'Circles', 'Spiral'])

    # Sliders for Learning Rate and Epochs in the sidebar
    learning_rate = st.sidebar.slider('Learning Rate', min_value=0.001, max_value=1.0, step=0.01, value=0.01)
    epochs = st.sidebar.slider('Epochs', min_value=10, max_value=10000, step=1, value=100)

    # input noise
    input_noise = st.sidebar.slider('Input Noise', min_value=0.0, max_value=1.0, step=0.01, value=0.1)

    # input number of samples
    num_samples = st.sidebar.slider('Number of Samples', min_value=100, max_value=10000, step=1, value=1000)

    # Get the data points and labels
    if dataset == 'Gaussian':
        data_points, labels = DataGenerator(num_samples, input_noise).generate_gauss_data()
    elif dataset == 'Moons':
        data_points, labels = DataGenerator(num_samples, input_noise).generate_moon_data()
    elif dataset == 'Spiral':
        data_points, labels = DataGenerator(num_samples, input_noise).generate_spiral_data()
    else:
        data_points, labels = DataGenerator(num_samples, input_noise).generate_circles_data()

    fig1 = plt.figure(figsize=(8, 6))
    plt.scatter(data_points[labels == 1, 0], data_points[labels == 1, 1], c='orange', label='Class 1')
    plt.scatter(data_points[labels == 0, 0], data_points[labels == 0, 1], c='purple', label='Class -1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Input Data')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig1)

    # Transform the data points based on the functions selected
    data_points_transformed = transformations(data_points, functions)

    # Create a checkbox with a label for MC Dropout
    mc_dropout = False
    mc_dropout = st.sidebar.checkbox('Enable MC Dropout', value=mc_dropout)

    # Button to generate the contour plot
    if st.button('Run Neural Network'):
        input_dim = len(functions)
        model = mlp_model(architecture_info, input_dim, learning_rate, mc_dropout)
        with st.spinner('Training the model and generating the plot...'):
            model.fit(data_points_transformed, labels, epochs=epochs, batch_size=10)

            # Show the contour plot on the main page using st.pyplot
            st.pyplot(generate_contour_plot(model, data_points, labels, functions, mc_dropout))


if __name__ == '__main__':
    main()
