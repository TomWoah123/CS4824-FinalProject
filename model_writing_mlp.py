import numpy as np
import pandas as pd


def rectified_linear_unit(z_values):
    """
    Code for the ReLu activation function

    :param z_values: An ndarray to represent the hidden layer
    :type: z_values: ndarray
    :return: the result of ReLu and the derivative of the ReLu activation function
    :rtype: tuple
    """
    relu = np.maximum(z_values, 0)
    relu_derivative = np.array(z_values > 0, dtype=int)
    return relu, relu_derivative


def softmax(output_values):
    """
    Code for the softmax activation function

    :param output_values: An ndarray to represent the output layer
    :type output_values: ndarray
    :return: the resulting vector of the output values
    :rtype: ndarray
    """
    return np.exp(output_values) / sum(np.exp(output_values))


def encoding(y_values):
    """
    Code used to encode the labels as an ndarray where each column represents the encoding of the y value as a vector

    :param y_values: The labels given for training
    :type: y_values: ndarray
    :return: An ndarray that represents the encodings of the labels
    :rtype: ndarray
    """
    one_hot_y = np.zeros((y_values.size, 10))
    one_hot_y[np.arange(y_values.size), y_values] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def multilayer_perceptron_generate_network(data_path):
    """
    This method trains a 3-layer Multilayer Perceptron model using training data that is supplied by the user.

    :param data_path: the path to the .csv file used for training the Multilayer Perceptron model
    :type: string
    """
    training_data = pd.read_csv(data_path)
    training_data_without_label = training_data.drop('label', axis=1)
    training_data_labels = training_data.label

    scale = lambda x: x / 255
    train_data_X = np.array(training_data_without_label).T
    train_data_X = scale(train_data_X)
    d, n = train_data_X.shape  # d = 784, n = 38000
    train_data_Y = np.array(training_data_labels)

    # Neural network with three layers: Input layer (784 nodes), Hidden layer (1056 nodes), and Output layer (10 nodes)
    alpha = 0.1
    num_hidden_layer_neurons = 1056
    input_to_hidden_layer_weights = np.random.rand(num_hidden_layer_neurons, d) - 0.5  # (1056 x 784)
    input_to_hidden_layer_bias_terms = np.random.rand(num_hidden_layer_neurons, 1) - 0.5  # (1056 x 1)
    hidden_to_output_layer_weights = np.random.rand(10, num_hidden_layer_neurons) - 0.5  # (10 x 1056)
    hidden_to_output_layer_bias_terms = np.random.rand(10, 1) - 0.5  # (10 x 1)
    iteration_accs = []

    for i in range(50):
        # Forward propagation first layer
        z = input_to_hidden_layer_weights.dot(
            train_data_X) + input_to_hidden_layer_bias_terms  # (1056 x 784) * (784 x 38000) = (1056 x 38000)
        activation, activation_derivative = rectified_linear_unit(z)  # 1056 x 38000

        # Forward propagation second layer
        output = hidden_to_output_layer_weights.dot(
            activation) + hidden_to_output_layer_bias_terms  # (10 x 1056) * (1056 x 38000) = (10 x 38000)
        output = softmax(output)  # 10 x 38000

        # Backpropagation second layer
        dz_output = 2 * (output - encoding(train_data_Y))  # 10 x 38000
        dweights_hidden_to_output = 1 / n * dz_output.dot(activation.T)  # (10 x 38000) * (38000 x 1056) = (10 x 1056)
        dbias_hidden_to_output = 1 / n * np.sum(dz_output, axis=1).reshape(10, 1)  # (10 x 1)

        # Backpropagation first layer
        dz_activation = hidden_to_output_layer_weights.T.dot(
            dz_output) * activation_derivative  # (1056 x 10) * (10 x 38000) = (1056 x 38000)
        dweights_input_to_hidden = 1 / n * dz_activation.dot(
            train_data_X.T)  # (1056 x 38000) * (38000 x 784) = (1056 x 784)
        dbias_input_to_hidden = 1 / n * np.sum(dz_activation, axis=1).reshape(num_hidden_layer_neurons, 1)  # (1056 x 1)

        # Updating parameters
        input_to_hidden_layer_weights = input_to_hidden_layer_weights - alpha * dweights_input_to_hidden  # (1056 x 784)
        input_to_hidden_layer_bias_terms = input_to_hidden_layer_bias_terms - alpha * dbias_input_to_hidden  # (1056 x 1)
        hidden_to_output_layer_weights = hidden_to_output_layer_weights - alpha * dweights_hidden_to_output  # (10 x 1056)
        hidden_to_output_layer_bias_terms = hidden_to_output_layer_bias_terms - alpha * dbias_hidden_to_output  # (10 x 1)

        # Get the accuracy
        predictions = np.argmax(output, axis=0)
        accuracy = np.sum(predictions == train_data_Y) / train_data_Y.size
        iteration_accs.append(accuracy)

        # Print the accuracy on every tenth iteration
        if i % 10 == 0:
            print(f"Training accuracy for iteration {i}:", "{:.5f}".format(accuracy))

    print("Finished training.....")
    np.savetxt("input_to_hidden_layer_weights.csv", input_to_hidden_layer_weights, delimiter=",")
    np.savetxt("input_to_hidden_layer_bias_terms.csv", input_to_hidden_layer_bias_terms, delimiter=",")
    np.savetxt("hidden_to_output_layer_weights.csv", hidden_to_output_layer_weights, delimiter=",")
    np.savetxt("hidden_to_output_layer_bias_terms.csv", hidden_to_output_layer_bias_terms, delimiter=",")


def mlp_output(data_path):
    """
    Test the trained Multilayer Perceptron model with testing data supplied by the user

    :param data_path: the path to the .csv file used for testing the pretrained Multilayer Perceptron model
    :type data_path: str
    :return: The accuracy of the model against the testing dataset
    :rtype: float
    """
    # Get the network from the .csv files
    input_to_hidden_layer_weights = np.array(pd.read_csv("input_to_hidden_layer_weights.csv", header=None))
    input_to_hidden_layer_bias_terms = np.array(pd.read_csv("input_to_hidden_layer_bias_terms.csv", header=None))
    hidden_to_output_layer_weights = np.array(pd.read_csv("hidden_to_output_layer_weights.csv", header=None))
    hidden_to_output_layer_bias_terms = np.array(pd.read_csv("hidden_to_output_layer_bias_terms.csv", header=None))

    # Scale the data given
    scale = lambda x: x / 255
    data = pd.read_csv(data_path)
    data_without_label = data.drop('label', axis=1)
    data_labels = data.label
    data = np.array(data_without_label).T
    data = scale(data)
    labels = np.array(data_labels)

    # Forward propagation first layer
    z_test = input_to_hidden_layer_weights.dot(data) + input_to_hidden_layer_bias_terms  # ( x 784) * (784 x 38000) = (512 x 38000)
    activation_test, activation_derivative_test = rectified_linear_unit(z_test)  # 512 x 38000

    # Forward propagation second layer
    test_output = hidden_to_output_layer_weights.dot(
        activation_test) + hidden_to_output_layer_bias_terms  # (10 x 512) * (512 x 38000) = (10 x 38000)
    test_output = softmax(test_output)  # 10 x 38000

    # Testing Data accuracy
    test_predictions = np.argmax(test_output, axis=0)
    test_accuracy = np.sum(test_predictions == labels) / labels.size
    return test_accuracy

