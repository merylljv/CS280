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
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


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


def dataset(validation_fraction, boost_minority=True):
    """ Partitions the dataset to training set and validation set. Writes 
        output to training_set.csv and validation_set.csv, respectively, with 
        corresponding labels at training_labels.csv and validation_labels.csv.

    Args:
        validation_fraction (float): The proportion of training data to set 
                            aside as validation set.
        boost_minority (bool): Will boost minority class to have training set
                            with equal instances per class if set to True.

    """

    data = pd.read_csv('data.csv', header=None)
    data_labels = pd.read_csv('data_labels.csv', names=['label'])
    data = pd.concat([data, data_labels], axis=1)
    
    class_data = data.groupby('label', as_index=False)
    data = class_data.apply(data_partition,
                            validation_fraction=validation_fraction)
    
    # training dataset
    training_set = data[~data.validation]
    # boost the number of minority class training examples
    if boost_minority:
        len_class = Counter(data.label)
        max_len = len_class.most_common(1)[0][1]
        set_labels = len_class.keys()
        for label in set_labels:
            class_training = training_set[training_set.label == label]
            num_boost = max_len - len(class_training)
            training_set = training_set.append(class_training.sample(num_boost,
                                                                     replace=True))
    training_set[['label']].to_csv('training_labels.csv', header=False,
                index=False)
    training_set = training_set.drop(['label', 'validation'], axis=1)
    training_set.to_csv('training_set.csv', header=False,
                  index=False)
    
    #validation dataset
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


def recall(labels, n_out, print_confusion=False):
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
    if print_confusion:
        print (confusion_matrix)
    TP = np.diag(confusion_matrix.values)
    TP_FN = confusion_matrix.sum(axis=0).values
    return np.mean(TP/TP_FN) * 100


def accuracy(labels):
    """ Computes accuracy of neural network.

    Args:
        label_comparison (dataframe): Contains desired and predicted labels.

    Returns:
        acc (float): Accuracy of neural network.

    """
    
    TP = len(labels[labels.predicted_label == labels.label])
    ALL = len(labels)
    acc = 100. * TP / ALL
    return acc


def training(learning_rate, max_epochs, h1_nodes='', h2_nodes=''):
    """ Trains the perceptron to calculate for weights in neural network.

    Args:
        learning_rate (float): Learning rate parameter.
        h1_nodes (int): Number of nodes in first hidden layer.
        h2_nodes (int): Number of nodes in second hidden layer.
        max_epochs (int): Maximum number of epochs.

    """

    # Training dataset
    train = pd.read_csv('training_set.csv', header=None)
    train_labels = pd.read_csv('training_labels.csv', names=['label'])
    # Number of input and output
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
    w_h1 = 0.1 * np.random.rand(h1_nodes, n_in)
    b_h1 = np.random.rand(h1_nodes, 1)
    w_h2 = 0.1 * np.random.rand(h2_nodes, h1_nodes)
    b_h2 = np.random.rand(h2_nodes, 1)
    w_out = 0.1 * np.random.rand(n_out, h2_nodes)
    b_out = np.random.rand(n_out, 1)

    # total error in epoch
    epoch_error = pd.DataFrame(columns=['epoch', 'net_err'])

    # Shuffled training dataset
    train_index = list(train.index)
    random.shuffle(train_index)

    for epoch in range(1, max_epochs+1):

        # training epoch
        for train_ins in train_index:
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
        net_err = 0.5* np.sum(np.sum(err**2)) / np.shape(err)[0]
        # accuracy and average recall
        label_comparison = predicted.reset_index()[['predicted_label']]
        label_comparison['label'] = validation_labels['label'].values
        ave_recall = recall(label_comparison, n_out)
        acc = accuracy(label_comparison)
        # error, accuracy and average recall per epoch
        epoch_error = epoch_error.append(pd.DataFrame({'epoch': [epoch],
                                                       'net_err': [net_err],
                                                       'recall': [ave_recall],
                                                       'acc': [acc]}))
#        print ('epoch %s: %s error, %s recall, %s accuracy' %(epoch,
#                                                        np.round(net_err, 4),
#                                                        np.round(ave_recall, 2),
#                                                        np.round(acc, 2)))
        # check average recall for stopping of training
        if ave_recall > 95:
            break
        
    return w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error


def svm_prediction():
    """ Trains and predict using SVM. Predicted labels are saved at
    predicted_other.csv.

    """

    # training set
    train = pd.read_csv('training_set.csv', header=None)
    train_labels = pd.read_csv('training_labels.csv', names=['label'])
    n_out = len(set(train_labels.label))
    # validation set
    validation = pd.read_csv('validation_set.csv', header=None)
    validation_labels = pd.read_csv('validation_labels.csv', names=['label'])
    # test set
    test = pd.read_csv('test_set.csv', header=None)
    
    # training the data
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train, train_labels)
    # evaluation of prediction using validation set
    label_comparison = pd.DataFrame({'predicted_label': clf.predict(validation)})
    label_comparison['label'] = validation_labels['label'].values
    ave_recall = recall(label_comparison, n_out, print_confusion=True)
    acc = accuracy(label_comparison)
    print ('%s recall, %s accuracy' %(np.round(ave_recall, 2), np.round(acc, 2)))

    # predicting test set
    predicted = pd.DataFrame({'label': clf.predict(test.values)})
    predicted.to_csv('predicted_other.csv', header=False, index=False)


def mlp_prediction():
    """ Trains and predict using MLP. Predicted labels are saved at
    predicted_ann.csv.

    """

    # training
    w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error = training(learning_rate=0.5,
                               max_epochs=300, h1_nodes = 21, h2_nodes = 15)

    # validation
    validation = pd.read_csv('validation_set.csv', header=None)
    validation_labels = pd.read_csv('validation_labels.csv', names=['label'])
    n_in = len(validation.columns)
    n_out = len(set(validation_labels.label))
    validation['grp'] = range(len(validation))
    validation_grp = validation.groupby('grp', as_index=False)
    predicted = validation_grp.apply(predict, n_in=n_in, w_h1=w_h1,
                                     b_h1=b_h1, w_h2=w_h2, b_h2=b_h2,
                                     w_out=w_out, b_out=b_out)
    # evaluation of prediction using validation set
    label_comparison = predicted.reset_index()[['predicted_label']]
    label_comparison['label'] = validation_labels['label'].values
    ave_recall = recall(label_comparison, n_out, print_confusion=True)
    acc = accuracy(label_comparison)
    print ('%s recall, %s accuracy' %(np.round(ave_recall, 2), np.round(acc, 2)))

    # predicting test set
    test = pd.read_csv('test_set.csv', header=None)
    test['grp'] = range(len(test))
    test_grp = test.groupby('grp', as_index=False)
    predicted = test_grp.apply(predict, n_in=n_in, w_h1=w_h1,
                                     b_h1=b_h1, w_h2=w_h2, b_h2=b_h2,
                                     w_out=w_out, b_out=b_out)
    predicted[['predicted_label']].to_csv('predicted_ann.csv',
                                          header=False, index=False)


def compare_boosted():
    """ Compares the error, average recall, and accuracy of training set with
    and without minority boost. Plots with respect to epoch number.

    """
    
    nns = {}
    for minority_boost in [False, True]:
        dataset(1/6., boost_minority=minority_boost)
        w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error = training(learning_rate=0.1,
                                   max_epochs=300, h1_nodes = 14, h2_nodes = 10)
        file_name = 'minority_boost_' + str(minority_boost)
        epoch_error.to_csv(file_name+'.csv', index=False)
        nns[str(minority_boost)] = epoch_error
        
    for minority_boost in [False, True]:
        epoch_error = nns.get(str(minority_boost))
        legend = 'with'
        if not minority_boost:
            legend += 'out'
        legend += ' minority boost'
        plt.plot(epoch_error.epoch, epoch_error.net_err, label=legend)
    plt.legend(loc=1)
    plt.xlabel('Epoch number')
    plt.ylabel('Prediction Error')
    plt.savefig('minority_boost_err.PNG')
    plt.close()

    for minority_boost in [False, True]:
        epoch_error = nns.get(str(minority_boost))
        legend = 'with'
        if not minority_boost:
            legend += 'out'
        legend += ' minority boost'
        plt.plot(epoch_error.epoch, epoch_error.acc, label=legend)
    plt.legend(loc=4)
    plt.xlabel('Epoch number')
    plt.ylabel('Percent Accuracy')
    plt.savefig('minority_boost_acc.PNG')
    plt.close()

    for minority_boost in [False, True]:
        epoch_error = nns.get(str(minority_boost))
        legend = 'with'
        if not minority_boost:
            legend += 'out'
        legend += ' minority boost'
        plt.plot(epoch_error.epoch, epoch_error.recall, label=legend)
    plt.legend(loc=4)
    plt.xlabel('Epoch number')
    plt.ylabel('Percent Average Recall')
    plt.savefig('minority_boost_recall.PNG')
    plt.close()


def compare_learning_rate(learning_rate_list):
    """ Compares the error, average recall, and accuracy of varying learning
    rate of the neural network. Plots with respect to epoch number.

    Args:
        learning_rate_list (list): List of learning rates.

    """
    
    nns = {}
    start_time = datetime.now()
    for learning_rate in learning_rate_list:
        w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error = training(learning_rate=0.1,
                                   max_epochs=300, h1_nodes = 7, h2_nodes = 5)
        file_name = 'lr' + str(learning_rate).replace('.', 'p') + '_h7_h5'
        epoch_error.to_csv(file_name+'.csv', index=False)
        nns[learning_rate] = epoch_error
        print ('runtime = %s' %str(datetime.now() - start_time))
        
    for learning_rate in learning_rate_list:
        epoch_error = nns.get(learning_rate)
        plt.plot(epoch_error.epoch, epoch_error.net_err,
                 label=r'$\mu$ = %s' %learning_rate)
    plt.legend(loc=1)
    plt.xlabel('Epoch number')
    plt.ylabel('Prediction Error')
    plt.savefig('learning_rate_err.PNG')
    plt.close()

    for learning_rate in learning_rate_list:
        epoch_error = nns.get(learning_rate)
        plt.plot(epoch_error.epoch, epoch_error.acc,
                 label=r'$\mu$ = %s' %learning_rate)
    plt.legend(loc=3)
    plt.xlabel('Epoch number')
    plt.ylabel('Percent Accuracy')
    plt.savefig('learning_rate_acc.PNG')
    plt.close()

    for learning_rate in learning_rate_list:
        epoch_error = nns.get(learning_rate)
        plt.plot(epoch_error.epoch, epoch_error.recall,
                 label=r'$\mu$ = %s' %learning_rate)
    plt.legend(loc=3)
    plt.xlabel('Epoch number')
    plt.ylabel('Percent Average Recall')
    plt.savefig('learning_rate_recall.PNG')
    plt.close()


def compare_hidden_layer(hidden_layers_list):
    """ Compares the error, average recall, and accuracy of varying number of
    neurons in the hidden layer of the neural network. Plots with respect to 
    epoch number.

    Args:
        hidden_layers_list (list): List of hidden layers of the form 
                                   [[h1, h2], [h1, h2], [h1, h2], ...].

    """
    
    nns = {}
    start_time = datetime.now()
    for hlist in hidden_layers_list:
        w_h1, b_h1, w_h2, b_h2, w_out, b_out, epoch_error = training(learning_rate=0.1,
                                   max_epochs=300, h1_nodes = hlist[0],
                                   h2_nodes = hlist[1])
        file_name = 'lr0p1_h' + str(hlist[0]) + '_h' + str(hlist[1])
        epoch_error.to_csv(file_name+'.csv', index=False)
        nns['-'.join(map(str, hlist))] = epoch_error
        print ('runtime = %s' %str(datetime.now() - start_time))
        
    for hlist in hidden_layers_list:
        harchi = '-'.join(map(str, hlist))
        epoch_error = nns.get(harchi)
        plt.plot(epoch_error.epoch, epoch_error.net_err, label=harchi)
    plt.legend(loc=1)
    plt.xlabel('Epoch number')
    plt.ylabel('Prediction Error')
    plt.savefig('hidden_layer_err.PNG')
    plt.close()

    for hlist in hidden_layers_list:
        harchi = '-'.join(map(str, hlist))
        epoch_error = nns.get(harchi)
        plt.plot(epoch_error.epoch, epoch_error.acc, label=harchi)
    plt.legend(loc=3)
    plt.xlabel('Epoch number')
    plt.ylabel('Percent Accuracy')
    plt.savefig('hidden_layer_acc.PNG')
    plt.close()

    for hlist in hidden_layers_list:
        harchi = '-'.join(map(str, hlist))
        epoch_error = nns.get(harchi)
        plt.plot(epoch_error.epoch, epoch_error.recall, label=harchi)
    plt.legend(loc=3)
    plt.xlabel('Epoch number')
    plt.ylabel('Percent Average Recall')
    plt.savefig('hidden_layer_recall.PNG')
    plt.close()


#################################################################################
if __name__ == '__main__':
    dataset(1/6.)
    start_time = datetime.now()
#    mlp_prediction()
#    mlp_end = datetime.now()
    svm_prediction()
    svm_end = datetime.now()
#    print('mlp runtime = %s' %str(mlp_end - start_time))
    print('svm runtime = %s' %str(svm_end - start_time))