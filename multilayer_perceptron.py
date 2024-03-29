import numpy as np
import pandas as pd

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

def multilayer_perceptron_output():
    training_data = pd.read_csv('./datasets/train.csv')
    training_data_without_label = training_data.drop('label', axis=1)
    training_data_labels = training_data.label
    testing_data = pd.read_csv('./datasets/test.csv')
    testing_data_without_label = testing_data.drop('label', axis=1)
    testing_data_labels = testing_data.label

    scale = lambda x: x / 255
    train_data_X = np.array(training_data_without_label).T
    train_data_X = scale(train_data_X)
    d, n = train_data_X.shape  # d = 784, n = 38000
    train_data_Y = np.array(training_data_labels)

    # Neural network with three layers: Input layer (784 nodes), Hidden layer (784 nodes), and Output layer (10 nodes)
    alpha = 0.1
    num_hidden_layer_neurons = 512
    input_to_hidden_layer_weights = np.random.rand(num_hidden_layer_neurons, d) - 0.5  # (784 x 784)
    input_to_hidden_layer_bias_terms = np.random.rand(num_hidden_layer_neurons, 1) - 0.5  # (784 x 1)
    hidden_to_output_layer_weights = np.random.rand(10, num_hidden_layer_neurons) - 0.5  # (10 x 784)
    hidden_to_output_layer_bias_terms = np.random.rand(10, 1) - 0.5  # (10 x 1)
    iteration_accs = []

    for i in range(50):
        # Forward propagation first layer
        z = input_to_hidden_layer_weights.dot(train_data_X) + input_to_hidden_layer_bias_terms  # (784 x 784) * (784 x 38000) = (784 x 38000)
        activation, activation_derivative = rectified_linear_unit(z)  # 512 x 38000

        # Forward propagation second layer
        output = hidden_to_output_layer_weights.dot(activation) + hidden_to_output_layer_bias_terms  # (10 x 784) * (784 x 38000) = (10 x 38000)
        output = softmax(output)  # 10 x 38000

        # Backpropagation second layer
        # dz_output = (output - encoding(train_data_Y)) / (output * (1 - output))  # 10 x 38000
        # dweights_hidden_to_output = 1 / n * (output - encoding(train_data_Y)).dot(activation.T)  # (10 x 38000) * (38000 x 784) = (10 x 784)
        dz_output = 2 * (output - encoding(train_data_Y))  # 10 x 38000
        dweights_hidden_to_output = 1 / n * dz_output.dot(activation.T)  # (10 x 38000) * (38000 x 512) = (10 x 512)
        dbias_hidden_to_output = 1 / n * np.sum(dz_output, axis=1).reshape(10, 1)  # (10 x 1)

        # Backpropagation first layer
        dz_activation = hidden_to_output_layer_weights.T.dot(dz_output) * activation_derivative  # (784 x 10) * (10 x 38000) = (784 x 38000)
        dweights_input_to_hidden = 1 / n * dz_activation.dot(train_data_X.T)  # (784 x 38000) * (38000 x 784) = (784 x 784)
        dbias_input_to_hidden = 1 / n * np.sum(dz_activation, axis=1).reshape(num_hidden_layer_neurons, 1)  # (512 x 1)

        # Updating parameters
        input_to_hidden_layer_weights = input_to_hidden_layer_weights - alpha * dweights_input_to_hidden  # (784 x 784)
        input_to_hidden_layer_bias_terms = input_to_hidden_layer_bias_terms - alpha * dbias_input_to_hidden  # (784 x 1)
        hidden_to_output_layer_weights = hidden_to_output_layer_weights - alpha * dweights_hidden_to_output  # (10 x 784)
        hidden_to_output_layer_bias_terms = hidden_to_output_layer_bias_terms - alpha * dbias_hidden_to_output  # (10 x 1)

        # Get the accuracy
        predictions = np.argmax(output, axis=0)
        accuracy = np.sum(predictions == train_data_Y) / train_data_Y.size
        iteration_accs.append(accuracy)

        # Print the accuracy on every tenth iteration
        if i % 10 == 0:
            print(f"Training accuracy for iteration {i}:", "{:.5f}".format(accuracy))


    print("Finished training!")
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
    # print(test_predictions, test_data_Y)
    print("Testing accuracy:", "{:.5f}".format(test_accuracy))
    return {'test_accuracy': test_accuracy, 'iteration_accs': iteration_accs}

def multilayer_perceptron_4_layer_output():
    iteration_accs = []
    training_data = pd.read_csv('./datasets/train.csv')
    training_data_without_label = training_data.drop('label', axis=1)
    training_data_labels = training_data.label
    testing_data = pd.read_csv('./datasets/test.csv')
    testing_data_without_label = testing_data.drop('label', axis=1)
    testing_data_labels = testing_data.label

    scale = lambda x: x / 255
    train_data_X = np.array(training_data_without_label).T
    train_data_X = scale(train_data_X)
    d, n = train_data_X.shape  # d = 784, n = 38000
    train_data_Y = np.array(training_data_labels)

    # Neural network with four layers: Input layer (784 nodes),
    # Hidden layers (784 nodes and 784 nodes), and Output layer (10 nodes)
    alpha = 0.10
    input_to_hidden_layer_weights = np.random.rand(784, d) * 2 - 1.0  # (784 x 784)
    input_to_hidden_layer_bias_terms = np.random.rand(784, 1) * 2 - 1.0  # (784 x 1)
    hidden_to_hidden_layer_weights = np.random.rand(784, 784) * 2 - 1.0  # (784 x 784)
    hidden_to_hidden_layer_bias_terms = np.random.rand(784, 1) * 2 - 1.0  # (784 x 1)
    hidden_to_output_layer_weights = np.random.rand(10, 784) * 2 - 1.0  # (10 x 784)
    hidden_to_output_layer_bias_terms = np.random.rand(10, 1) * 2 - 1.0  # (10 x 1)

    for i in range(50):
        # Forward propagation first layer
        z_one = input_to_hidden_layer_weights.dot(
            train_data_X) + input_to_hidden_layer_bias_terms  # (784 x 784) * (784 x 38000) = (784 x 38000)
        activation_one, activation_derivative_one = rectified_linear_unit(z_one)  # 784 x 38000

        # Forward propagation second layer
        z_two = hidden_to_hidden_layer_weights.dot(
            activation_one) + hidden_to_hidden_layer_bias_terms  # (784 x 784) * (784 x 38000) = (784 x 38000)
        activation_two, activation_derivative_two = rectified_linear_unit(z_two)  # (784 x 38000)

        # Forward propagation third layer
        output = hidden_to_output_layer_weights.dot(
            activation_two) + hidden_to_output_layer_bias_terms  # (10 x 784) * (784 x 38000) = (10 x 38000)
        output = softmax(output)  # 10 x 38000

        # Backpropagation third layer
        dz_output = 2 * (output - encoding(train_data_Y))  # 10 x 38000
        dweights_hidden_to_output = 1 / n * dz_output.dot(activation_two.T)  # (10 x 38000) * (38000 x 784) = (10 x 784)
        dbias_hidden_to_output = 1 / n * np.sum(dz_output, axis=1).reshape(10, 1)  # (10 x 1)

        # Backpropagation second layer
        dz_activation_two = hidden_to_output_layer_weights.T.dot(
            dz_output) * activation_derivative_two  # (784 x 10) * (10 x 38000) = (784 x 38000)
        dweights_hidden_to_hidden = 1 / n * dz_activation_two.dot(
            activation_one.T)  # (784 x 38000) * (38000 x 784) = (784 x 784)
        dbias_hidden_to_hidden = 1 / n * np.sum(dz_activation_two, axis=1).reshape(784, 1)  # (784 x 1)

        # Backpropagation first layer
        dz_activation_one = hidden_to_hidden_layer_weights.T.dot(
            dz_activation_two) * activation_derivative_one  # (784 x 784) * (784 x 38000) = (784 x 38000)
        dweights_input_to_hidden = 1 / n * dz_activation_one.dot(
            train_data_X.T)  # (784 x 38000) * (38000 x 784) = (784 x 784)
        dbias_input_to_hidden = 1 / n * np.sum(dz_activation_one, axis=1).reshape(784, 1)  # (784 x 1)

        # Updating parameters
        input_to_hidden_layer_weights = input_to_hidden_layer_weights - alpha * dweights_input_to_hidden  # (784 x 784)
        input_to_hidden_layer_bias_terms = input_to_hidden_layer_bias_terms - alpha * dbias_input_to_hidden  # (784 x 1)
        hidden_to_hidden_layer_weights = hidden_to_hidden_layer_weights - alpha * dweights_hidden_to_hidden  # (784 x 784)
        hidden_to_hidden_layer_bias_terms = hidden_to_hidden_layer_bias_terms - alpha * dbias_hidden_to_hidden  # (784 x 1)
        hidden_to_output_layer_weights = hidden_to_output_layer_weights - alpha * dweights_hidden_to_output  # (10 x 784)
        hidden_to_output_layer_bias_terms = hidden_to_output_layer_bias_terms - alpha * dbias_hidden_to_output  # (10 x 1)

        predictions = np.argmax(output, axis=0)
        accuracy = np.sum(predictions == train_data_Y) / train_data_Y.size
        iteration_accs.append(accuracy)

        # Print the accuracy on every tenth iteration
        if i % 10 == 0:
            print(f"Training accuracy for iteration {i}:", "{:.5f}".format(accuracy))

    print("Finished training.......")
    test_data_X = np.array(testing_data_without_label).T
    test_data_X = scale(test_data_X)
    test_data_Y = np.array(testing_data_labels)

    # Forward propagation first layer
    z_one_test = input_to_hidden_layer_weights.dot(
        test_data_X) + input_to_hidden_layer_bias_terms  # (512 x 784) * (784 x 38000) = (512 x 38000)
    activation_one_test, activation_derivative_one_test = rectified_linear_unit(z_one_test)  # 512 x 38000

    # Forward propagation second layer
    z_two_test = hidden_to_hidden_layer_weights.dot(
        activation_one_test) + hidden_to_hidden_layer_bias_terms  # (128 x 512) * (512 x 38000) = (128 x 38000)
    activation_two_test, activation_derivative_two_test = rectified_linear_unit(z_two_test)  # (128 x 38000)

    # Forward propagation third layer
    test_output = hidden_to_output_layer_weights.dot(
        activation_two_test) + hidden_to_output_layer_bias_terms  # (10 x 128) * (128 x 38000) = (10 x 38000)
    test_output = softmax(test_output)  # 10 x 38000

    # Testing Data accuracy
    test_predictions = np.argmax(test_output, axis=0)
    test_accuracy = np.sum(test_predictions == test_data_Y) / test_data_Y.size
    # print(test_predictions, test_data_Y)
    print("Testing accuracy:", "{:.5f}".format(test_accuracy))
    return {'test_accuracy': test_accuracy, 'iteration_accs': iteration_accs}
