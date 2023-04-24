import numpy as np
import pandas as pd
import csv

def naive_bayes_generate_model(data_path):
    """Train naive Bayes parameters from the given training dataset.

    :param data_path: the path to the .csv file used for training a Naive Bayes classifier
    :type data_path: string
    """
    training_data = pd.read_csv(data_path)
    training_data_without_label = training_data.drop('label', axis=1)
    training_data_labels = training_data.label
    scale = lambda x: 0 if x < 15 else 1
    scale = np.vectorize(scale)
    train_data_X = np.array(training_data_without_label).T
    train_data = scale(train_data_X)
    train_labels = np.array(training_data_labels)

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


def naive_bayes_predict_using_model(data_path):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data_path: the path to the .csv file used for testing the pretrained Naive Bayes model
    :type data: string
    :return: The accuracy of the model using the testing dataset
    :rtype: float
    """
    data = pd.read_csv(data_path)
    data_without_label = data.drop('label', axis=1)
    data_labels = data.label

    scale = lambda x: 0 if x < 15 else 1
    scale = np.vectorize(scale)
    data = np.array(data_without_label).T
    data = scale(data)
    labels = np.array(data_labels)

    cond = pd.read_csv("conditional_log_prob.csv", header=None)
    prior = pd.read_csv("prior_log_prob.csv", header=None)
    conditional_log_prob = np.array(cond)
    prior_log_prob = np.array(prior)

    log_class_probs = conditional_log_prob.T.dot(data) \
        + np.log(1 - np.exp(conditional_log_prob)).T.dot(1 - data) \
        + prior_log_prob.reshape((prior_log_prob.size, 1))

    predictions = np.argmax(log_class_probs, axis=0).ravel()
    accuracy = np.sum(predictions == labels) / labels.size
    return accuracy


