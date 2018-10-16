# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 16:50:41 2018

@author: Meryll

Works with Python 2.7 and Windows 10
Uses pandas version 19.2

Assumptions:
    data.csv, data_labels.csv, test_set.csv has same path as this python script.
    data.csv contains input used in training of neural network.
    data_labels.csv contains corresponding desired output (natural number).
    test_set.csv has same format as data.csv.
"""

from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_partition(dataset, validation_fraction):
    """ Adds column to categorize per class if dataset is part of training set 
    or validation set.
    
    Args:
        dataset (dataframe): Training data to be categorized.
        validation_fraction (float): The proportion of training data to set 
                                     aside as validation set.
                            
    Returns:
        dataset (dataframe): Training data with 'validation' column; 
                             True if part of validation set, False otherwise.

    """

    validation_length = int(len(dataset) * validation_fraction)
    dataset_index = dataset.index
    validation_index = np.random.choice(dataset_index, validation_length,
                                        replace=False)
    dataset['validation'] = dataset.index.isin(validation_index)
    return dataset


def dataset(validation_fraction):
    """ Partitions the dataset to training set and validation set. Writes 
        output to training_set.csv and validation_set.csv, respectively, with 
        corresponding labels at training_labels.csv and validation_labels.csv.

    Args:
        validation_fraction (float): The proportion of training data to set 
                            aside as validation set.

    """

    data = pd.read_csv('data.csv', header=None)
    data_labels = pd.read_csv('data_labels.csv', names=['label'])
    data = pd.concat([data, data_labels], axis=1)
    
#    len_class = Counter(data.label)
#    max_len = len_class.most_common(1)[0][1]
    
    training_set = pd.DataFrame()
    validation_set = pd.DataFrame()
    class_data = data.groupby('label', as_index=False)
    data = class_data.apply(data_partition,
                            validation_fraction=validation_fraction)
    
    training_set = data[~data.validation]
    training_set[['label']].to_csv('training_labels.csv', header=False,
                index=False)
    training_set = training_set.drop(['label', 'validation'], axis=1)
    training_set.to_csv('training_set.csv', header=False,
                  index=False)
    
    validation_set = data[data.validation]
    validation_set[['label']].to_csv('validation_labels.csv', header=False,
                  index=False)
    validation_set = validation_set.drop(['label', 'validation'], axis=1)
    validation_set.to_csv('validation_set.csv', header=False,
                  index=False)


def predict(dataset, n_in, w_h1, b_h1, w_h2, b_h2, w_out, b_out):
    """ Predicts output given weights of neural network.

    Args:
        dataset (dataframe): Contains input of neural network.
        w_h1 (array): Weigths of first hidden layer.
        b_h1 (array): Bias of first hidden layer.
        w_h2 (array): Weigths of second hidden layer.
        b_h2 (array): Bias of second hidden layer.

    Returns:
        out (dataframe): Predicted output of neural network.

    """

    # read data
    index = dataset['grp'].values[0]
    x_in = dataset.drop(['grp'], axis=1).values[0].reshape(n_in, 1)
    # hidden layer 1
    v_h1 = np.matmul(w_h1, x_in) + b_h1
    y_h1 = 1. / (1 + np.exp(-v_h1))
    # hidden layer 2
    v_h2 = np.matmul(w_h2, y_h1) + b_h2
    y_h2 = 1. / (1 + np.exp(-v_h2))
    # output layer
    v_out = np.matmul(w_out, y_h2) + b_out
    out = 1. / (1 + np.exp(-v_out))
    # predicted label
    label = list(out).index(max(out)) + 1
    
    predicted = pd.DataFrame({'grp': [index], 'out': [out],
                              'predicted_label': [label]})
    
    return predicted


def recall(labels, n_out):
    """ Predicts output given weights of neural network.

    Args:
        label_comparison (dataframe): Contains desired and predicted labels.
        n_out (int): Number of unique labels.

    Returns:
        ave_recall (float): Average recall of neural network.

    """
    
    set_labels = range(1, n_out+1)
    confusion_matrix = pd.DataFrame(index=set_labels,
                                    columns=set_labels)
    confusion_matrix.index.names = ['predicted']
    for actual in set_labels:
        for predicted in set_labels:
            count = len(labels[(labels.predicted_label == predicted) & \
                               (labels.label == actual)])
            confusion_matrix.loc[[predicted], [actual]] = count
    print confusion_matrix
    TP = np.diag(confusion_matrix.values)
    TP_FN = confusion_matrix.sum(axis=0).values
    print TP/TP_FN
    return np.mean(TP/TP_FN) * 100


def training(learning_rate, max_epochs, h1_nodes='', h2_nodes=''):
    """ Trains the perceptron to calculate for weights in neural network.

    Args:
        learning_rate (float): Learning rate parameter.
        h1_nodes (int): Number of nodes in first hidden layer.
        h2_nodes (int): Number of nodes in second hidden layer.
        max_epochs (int): Maximum number of epochs.

    """

    # Shuffled training dataset
    train = pd.read_csv('training_set.csv', header=None)
    train_labels = pd.read_csv('training_labels.csv', names=['label'])
    train = pd.concat([train, train_labels], axis=1)
    train = train.sample(frac=1)
    train_labels = train[['label']]
    train = train.drop(['label'], axis=1)
    #number of input and output
    n_in = len(train.columns) 
    n_out = len(set(train_labels.label))
    train_labels['out'] = train_labels['label'].apply(lambda x: np.array(map(int,
                                              '0'*(x-1) + '1' + '0'*(n_out-x))))
    
    # Set default number of nodes in hidden layers
    half_tot_nodes = np.sqrt((n_out + 2) * n_in)
    if h1_nodes == '':
        h1_nodes = int(np.round(((n_out + 4) / (n_out + 2.)) * half_tot_nodes, 0))
    if h2_nodes == '':
        h2_nodes = int(np.round((n_out / (n_out + 2.)) * half_tot_nodes, 0))
    
    # Validation dataset
    validation = pd.read_csv('validation_set.csv', header=None)
    validation_labels = pd.read_csv('validation_labels.csv', names=['label'])
    validation_labels['out'] = validation_labels['label'].apply(lambda x: np.array(map(int,
                                              '0'*(x-1) + '1' + '0'*(n_out-x))))
    validation_labels['out'] = validation_labels['out'].apply(lambda x: x.reshape(n_out, 1))

    # Initialize weights
    w_h1 = np.random.rand(h1_nodes, n_in)
    b_h1 = np.random.rand(h1_nodes, 1)
    w_h2 = np.random.rand(h2_nodes, h1_nodes)
    b_h2 = np.random.rand(h2_nodes, 1)
    w_out = np.random.rand(n_out, h2_nodes)
    b_out = np.random.rand(n_out, 1)

    # total error in epoch
    epoch_error = pd.DataFrame(columns=['epoch', 'tot_err'])

    for epoch in range(max_epochs):

        # training epoch
        for train_ins in train.index:
            # read data
            x_in = train[train.index == train_ins].values.reshape(n_in, 1)
            d_out = train_labels[train_labels.index == train_ins]['out'].values[0].reshape(n_out, 1)
        # forward pass
            # hidden layer 1
            v_h1 = np.matmul(w_h1, x_in) + b_h1
            y_h1 = 1. / (1 + np.exp(-v_h1))
            # hidden layer 2
            v_h2 = np.matmul(w_h2, y_h1) + b_h2
            y_h2 = 1. / (1 + np.exp(-v_h2))
            # output layer
            v_out = np.matmul(w_out, y_h2) + b_out
            out = 1. / (1 + np.exp(-v_out))
        # error propagation
            err = d_out - out
            # compute gradient in output layer
            delta_out = err * out * (1 - out)
            # compute gradient in hidden layer 2
            delta_h2 = y_h2 * (1-y_h2) * np.matmul(w_out.transpose(), delta_out)
            # compute gradient in hidden layer 1
            delta_h1 = y_h1 * (1-y_h1) * np.matmul(w_h2.transpose(), delta_h2)
            # update weights and biases in output layer 
            w_out = w_out + learning_rate * np.matmul(delta_out, y_h2.transpose())
            b_out = b_out + learning_rate * delta_out
            # update weights and biases in hidden layer 2
            w_h2 = w_h2 + learning_rate * np.matmul(delta_h2, y_h1.transpose())
            b_h2 = b_h2 + learning_rate * delta_h2
            # update weights and biases in hidden layer 1
            w_h1 = w_h1 + learning_rate * np.matmul(delta_h1, x_in.transpose())
            b_h1 = b_h1 + learning_rate * delta_h1

        # validation
        validation['grp'] = range(len(validation))
        validation_grp = validation.groupby('grp', as_index=False)
        predicted = validation_grp.apply(predict, n_in=n_in, w_h1=w_h1,
                                         b_h1=b_h1, w_h2=w_h2, b_h2=b_h2,
                                         w_out=w_out, b_out=b_out)
        # prediction error
        actual_labels = validation_labels['out'].values.reshape(len(predicted), 1)
        predicted_labels = predicted['out'].values.reshape(len(predicted), 1)
        err = actual_labels - predicted_labels
        tot_err = np.sum(np.sum(err**2))
        # precision
        label_comparison = predicted.reset_index()[['predicted_label']]
        label_comparison['label'] = validation_labels['label'].values
        ave_recall = recall(label_comparison, n_out)
        # error and recall per epoch
        epoch_error = epoch_error.append(pd.DataFrame({'epoch': [epoch],
                                                       'tot_err': [tot_err],
                                                       'recall': [ave_recall]}))
        print ('epoch %s: %s error, %s recall' %(epoch, tot_err, ave_recall))
        ########### check precision for early stopping ############
        if tot_err < 1:
            break
        
    return w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error


    #for each epoch
    #    for each training data instance
    #        propagate error through the network
    #        adjust the weights
    #        calculate the accuracy over training data
    #    for each validation data instance
    #        calculate the accuracy over the validation data
    #    if the threshold validation precision is met
    #        exit training
    #    else
    #        continue training


#Precision for one class 'A' is TP_A / (TP_A + FP_A) as in the mentioned article. Now you can calculate average precision of a model. There are a few ways of averaging (micro, macro, weighted), well explained here:
#
#'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance (...)



#################################################################################
if __name__ == '__main__':
    start_time = datetime.now()

    dataset(1/6.)
    
    w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error = training(learning_rate=0.5,
                                                                 max_epochs=100,
                                                                 h1_nodes=7,
                                                                 h2_nodes=5)
    plt.plot(epoch_error.epoch, epoch_error.tot_err)
    plt.plot(epoch_error.epoch, epoch_error.recall)

#    n_in = 354
#    n_out = 8
#    test = pd.read_csv('validation_set.csv', header=None)
#    test_labels = pd.read_csv('validation_labels.csv', names=['label'])
#    test_labels['out'] = test_labels['label'].apply(lambda x: np.array(map(int,
#                                              '0'*(x-1) + '1' + '0'*(n_out-x))))
#    test_labels['out'] = test_labels['out'].apply(lambda x: x.reshape(n_out, 1))
#    test['grp'] = range(len(test))
#    test_grp = test.groupby('grp', as_index=False)
#    predicted = test_grp.apply(predict, n_in=n_in, w_h1=w_h1, b_h1=b_h1,
#                               w_h2=w_h2, b_h2=b_h2, w_out=w_out, b_out=b_out)
#    label_comparison = predicted.reset_index()[['predicted_label']]
#    label_comparison['label'] = test_labels['label'].values

    runtime = datetime.now() - start_time
    print ('runtime: ' + str(runtime))