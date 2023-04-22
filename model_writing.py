import numpy as np
import pandas as pd
import csv

def naive_bayes_generate_model(train_data, train_labels):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """

    labels = np.unique(train_labels)

    d, n = train_data.shape
    num_classes = labels.size

    counts = np.zeros((d, num_classes))
    class_total = np.zeros(num_classes)

    for c in range(num_classes):
        counts[:, c] = train_data[:, train_labels == c].sum(1).ravel()
        class_total[c] = np.count_nonzero(train_labels == c)

    prior_log_prob = np.log(class_total) - np.log(n)
    conditional_log_prob = np.log(counts + 1) - np.log(class_total + 2).T

    np.savetxt("conditional_log_prob.csv", conditional_log_prob, delimiter=",")
    np.savetxt("prior_log_prob.csv", prior_log_prob, delimiter=",")

def naive_bayes_predict_using_model(data):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    cond = pd.read_csv("conditional_log_prob.csv", header=None)
    prior = pd.read_csv("prior_log_prob.csv", header=None)
    conditional_log_prob = np.array(cond)
    prior_log_prob = np.array(prior)

    log_class_probs = conditional_log_prob.T.dot(data) \
        + np.log(1 - np.exp(conditional_log_prob)).T.dot(1 - data) \
        + prior_log_prob.reshape((prior_log_prob.size, 1))

    labels = np.argmax(log_class_probs, axis=0).ravel()
    return labels


training_data = pd.read_csv('./datasets/train.csv')
training_data_without_label = training_data.drop('label', axis=1)
training_data_labels = training_data.label
testing_data = pd.read_csv('./datasets/test.csv')
testing_data_without_label = testing_data.drop('label', axis=1)
testing_data_labels = testing_data.label

scale = lambda x: 0 if x < 15 else 1
scale = np.vectorize(scale)
train_data_X = np.array(training_data_without_label).T
train_data_X = scale(train_data_X)
d, n = train_data_X.shape  # d = 784, n = 38000
train_data_Y = np.array(training_data_labels)
naive_bayes_generate_model(train_data_X, train_data_Y)
predictions = naive_bayes_predict_using_model(train_data_X)
train_accuracy = np.sum(predictions == train_data_Y) / train_data_Y.size
print("Training data accuracy: ", "{:.5f}".format(train_accuracy))
