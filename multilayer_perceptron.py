import numpy as np
import pandas as pd

labels = list(range(10))
training_data = pd.read_csv('./datasets/train.csv')
training_data_without_label = training_data.drop('label', axis=1)
training_data_labels = training_data.label
testing_data = pd.read_csv('./datasets/test.csv')
testing_data_without_label = testing_data.drop('label', axis=1)
testing_data_labels = testing_data.label

train_data_X = np.array(training_data_without_label).T
train_data_Y = np.array(training_data_labels)


def init_network():
    input_layer_weights = np.random.rand(10, 784) - 0.5
    input_layer_bias_terms = np.random.rand(10, 1) - 0.5
    hidden_layer_weights = np.random.rand(10, 10) - 0.5
    hidden_layer_bias_terms = np.random.rand(10, 1) - 0.5
    return input_layer_weights, input_layer_bias_terms, hidden_layer_weights, hidden_layer_bias_terms


def sigmoid(z):
    return 1 / (1 + np.exp(z))


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(output):
    return np.exp(output) / np.sum(np.exp(output))


def forward_propagation(input_layer_weights, input_layer_bias_terms, hidden_layer_weights, hidden_layer_bias_terms,
                        data):
    z_1 = input_layer_weights.dot(data) + input_layer_bias_terms
    activation_1 = sigmoid(z_1)
    z_2 = hidden_layer_weights.dot(activation_1) + hidden_layer_bias_terms
    activation_2 = softmax(z_2)
    return z_1, activation_1, z_2, activation_2


def backward_propagation(z_1, activation_1, z_2, activation_2, input_layer_weights, hidden_layer_weights, x, y):
    output_vector = np.zeros((y.size, 10))
    output_vector[np.arange(y.size), y] = 1
    output_vector = output_vector.T
    dz_2 = activation_2 - output_vector
    dw_2 = 1 / y.size * dz_2.dot(activation_1.T)
    db_2 = 1 / y.size * np.sum(dz_2)
    dz_1 = hidden_layer_weights.T.dot(dz_2) * derivative_sigmoid(z_1)
    dw_1 = 1 / y.size * dz_1.dot(x.T)
    db_1 = 1 / y.size * np.sum(dz_1)
    return dw_1, db_1, dw_2, db_2


def gradient_descent(x, y):
    alpha = 0.1
    input_layer_weights, input_layer_bias_terms, hidden_layer_weights, hidden_layer_bias_terms = init_network()
    for i in range(500):
        z_1, activation_1, z_2, activation_2 = forward_propagation(input_layer_weights, input_layer_bias_terms,
                                                                   hidden_layer_weights, hidden_layer_bias_terms, x)
        dw_1, db_1, dw_2, db_2 = backward_propagation(z_1, activation_1, z_2, activation_2,
                                                      input_layer_weights, hidden_layer_weights, x, y)
        input_layer_weights = input_layer_weights - alpha * dw_1
        input_layer_bias_terms = input_layer_bias_terms - alpha * db_1
        hidden_layer_weights = hidden_layer_weights - alpha * dw_2
        hidden_layer_bias_terms = hidden_layer_bias_terms - alpha * db_2
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = np.argmax(activation_2, axis=0)
            print(predictions, y)
            accuracy = np.sum(predictions == y) / y.size
            print(accuracy)
    return input_layer_weights, input_layer_bias_terms, hidden_layer_weights, hidden_layer_bias_terms


gradient_descent(train_data_X, train_data_Y)
