import numpy as np
import pandas as pd

training_data = pd.read_csv('./datasets/train.csv')
training_data_without_label = training_data.drop('label', axis=1)
training_data_labels = training_data.label
testing_data = pd.read_csv('./datasets/test.csv')
testing_data_without_label = testing_data.drop('label', axis=1)
testing_data_labels = testing_data.label


def rectified_linear_unit(z_values):
    relu = np.maximum(z_values, 0)
    relu_derivative = np.array(z_values > 0, dtype=int)
    return relu, relu_derivative


def softmax(output_values):
    return np.exp(output_values) / sum(np.exp(output_values))


def encoding(y_values):
    one_hot_y = np.zeros((y_values.size, 10))
    one_hot_y[np.arange(y_values.size), y_values] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


scale = lambda x: x / 255
train_data_X = np.array(training_data_without_label).T
train_data_X = scale(train_data_X)
d, n = train_data_X.shape  # d = 784, n = 38000
train_data_Y = np.array(training_data_labels)

# Neural network with three layers: Input layer (784 nodes), Hidden layer (512 nodes), and Output layer (10 nodes)
alpha = 0.1
input_to_hidden_layer_weights = np.random.rand(512, d) - 0.5  # (512 x 784)
input_to_hidden_layer_bias_terms = np.random.rand(512, 1) - 0.5  # (512 x 1)
hidden_to_output_layer_weights = np.random.rand(10, 512) - 0.5  # (10 x 512)
hidden_to_output_layer_bias_terms = np.random.rand(10, 1) - 0.5  # (10 x 1)

for i in range(50):
    # Forward propagation first layer
    z = input_to_hidden_layer_weights.dot(train_data_X) + input_to_hidden_layer_bias_terms  # (512 x 784) * (784 x 38000) = (512 x 38000)
    activation, activation_derivative = rectified_linear_unit(z)  # 512 x 38000

    # Forward propagation second layer
    output = hidden_to_output_layer_weights.dot(activation) + hidden_to_output_layer_bias_terms  # (10 x 512) * (512 x 38000) = (10 x 38000)
    output = softmax(output)  # 10 x 38000

    # Backpropagation second layer
    # dz_output = (output - encoding(train_data_Y)) / (output * (1 - output))  # 10 x 38000
    # dweights_hidden_to_output = 1 / n * (output - encoding(train_data_Y)).dot(activation.T)  # (10 x 38000) * (38000 x 512) = (10 x 512)
    dz_output = 2 * (output - encoding(train_data_Y))  # 10 x 38000
    dweights_hidden_to_output = 1 / n * dz_output.dot(activation.T)  # (10 x 38000) * (38000 x 512) = (10 x 512)
    dbias_hidden_to_output = 1 / n * np.sum(dz_output, axis=1).reshape(10, 1)  # (10 x 1)

    # Backpropagation first layer
    dz_activation = hidden_to_output_layer_weights.T.dot(dz_output) * activation_derivative  # (512 x 10) * (10 x 38000) = (512 x 38000)
    dweights_input_to_hidden = 1 / n * dz_activation.dot(train_data_X.T)  # (512 x 38000) * (38000 x 784) = (512 x 784)
    dbias_input_to_hidden = 1 / n * np.sum(dz_activation, axis=1).reshape(512, 1)  # (512 x 1)

    # Updating parameters
    input_to_hidden_layer_weights = input_to_hidden_layer_weights - alpha * dweights_input_to_hidden  # (512 x 784)
    input_to_hidden_layer_bias_terms = input_to_hidden_layer_bias_terms - alpha * dbias_input_to_hidden  # (512 x 1)
    hidden_to_output_layer_weights = hidden_to_output_layer_weights - alpha * dweights_hidden_to_output  # (10 x 512)
    hidden_to_output_layer_bias_terms = hidden_to_output_layer_bias_terms - alpha * dbias_hidden_to_output  # (10 x 1)

    if i % 10 == 0:
        predictions = np.argmax(output, axis=0)
        accuracy = np.sum(predictions == train_data_Y) / train_data_Y.size
        print("Iteration: ", i)
        print(predictions, train_data_Y)
        print("Accuracy: ", accuracy)


print("Finished training.......")
test_data_X = np.array(testing_data_without_label).T
test_data_X = scale(test_data_X)
test_data_Y = np.array(testing_data_labels)

# Forward propagation first layer
z_test = input_to_hidden_layer_weights.dot(test_data_X) + input_to_hidden_layer_bias_terms  # (512 x 784) * (784 x 38000) = (512 x 38000)
activation_test, activation_derivative_test = rectified_linear_unit(z_test)  # 512 x 38000

# Forward propagation second layer
test_output = hidden_to_output_layer_weights.dot(activation_test) + hidden_to_output_layer_bias_terms  # (10 x 512) * (512 x 38000) = (10 x 38000)
test_output = softmax(test_output)  # 10 x 38000

# Testing Data accuracy
test_predictions = np.argmax(test_output, axis=0)
test_accuracy = np.sum(test_predictions == test_data_Y) / test_data_Y.size
print(test_predictions, test_data_Y)
print("Accuracy: ", test_accuracy)
