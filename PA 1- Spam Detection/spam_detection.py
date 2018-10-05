# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:15:20 2018

@author: Meryll
"""

from collections import Counter
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         'trec06p-cs280\\data'))


def create_vocabulary():
    """ Creates vocabulary V of unique words in the training data. Saves output
    to full_vocabulary.csv.
    
    """

    all_words = []
    
    # Compiles all words in the training set in a list
    for dir_path, dir_names, file_list in os.walk(data_path):
        try:
            if int(dir_path[len(dir_path)-3::]) <= 70:
                for file_name in file_list:
                    cur_file = os.path.join(dir_path, file_name)
                    original_string = open(cur_file, 'r').read()
                    new_string = original_string.replace("'", '').lower()
                    doc_word = re.findall('[a-z]+', new_string)
                    all_words += doc_word
        except:
            pass
    
    # Frequency of each word in the training set
    word_count = Counter(all_words)
    df = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
    df = df.rename(columns={'index':'word', 0:'count'})
    df = df.dropna()
    df['length'] = df['word'].apply(lambda x: len(x))
    df = df[df.length > 1]
    df = df.sort_values('count', ascending=False)
    df.to_csv('full_vocabulary.csv', index=False)
    
    
def parse_doc(doc, word_presence):
    """ Checks presence of each vocabulary word in each document.
    
    Args:
        doc (dataframe): Contains file path of the document.
        word_presence (dataframe): Contains words in the vocabulary.

    """

    doc_path = doc['data'].values[0]
    file_path = os.path.abspath(os.path.join(data_path, doc_path))
    doc_text = open(file_path, 'r').read().replace("'", '').lower()
    doc_word = re.findall('[a-z]+', doc_text)
    word_presence[doc_path] = word_presence['word'].apply(lambda x: int(x in \
                                                          doc_word))


def word_presence_matrix(num_vocab=10000):
    """ Creates matrix that contains presence of each vocabulary word in 
    each document. Saves output to spam_train.csv, ham_train.csv,
    spam_test.csv, ham_test.csv.
    
    Args:
        num_vocab (int): Preferred number of words in the vocabulary.
    
    """
                                             
    vocabulary = pd.read_csv('full_vocabulary.csv')[0:num_vocab]
    
    # Classifies each document to class spam or train and to training or test set
    labels = pd.read_csv(os.path.abspath(os.path.join(data_path, '../labels')),
                         sep=' ', names=['cls', 'data'])
    labels['folder'] = labels['data'].apply(lambda x: int(x[8:11]))    
    train = labels[labels.folder <= 70]
    test = labels[labels.folder > 70]
    spam_train = train[train.cls == 'spam']
    ham_train = train[train.cls == 'ham']
    spam_test = test[test.cls == 'spam']
    ham_test = test[test.cls == 'ham']
    dct = dict({'spam_train': spam_train, 'ham_train': ham_train,
                'spam_test': spam_test, 'ham_test': ham_test})

    # Checks word presence per classification per set
    for i in dct.keys():
        per_doc = dct.get(i).groupby('data', as_index=False)
        word_presence = vocabulary[['word']]
        per_doc.apply(parse_doc, word_presence=word_presence)
        word_presence.to_csv(i + '.csv', index=False)
    

def probabilities():
    """ Checks prior probability of each word in the class (spam or ham).
    
    Returns:
        prob (dataframe): Contains class conditional likelihood.
        num_spam (int): Number of mail in the training set classified as spam.
        num_ham (int): Number of mail in the training set classified as ham.

    """

    # Probability of each word in the class
    spam_train = pd.read_csv('spam_train.csv')
    ham_train = pd.read_csv('ham_train.csv')
    spam_train = spam_train.set_index('word')
    ham_train = ham_train.set_index('word')
    num_spam = len(spam_train.columns)
    num_ham = len(ham_train.columns)
    spam_train['spam_prob'] = spam_train.sum(axis=1) / num_spam
    ham_train['ham_prob'] = ham_train.sum(axis=1) / num_ham
    
    # Prior probability of spam and ham
    prob_spam = 1.0 * num_spam / (num_spam + num_ham)
    prob_ham = 1.0 * num_ham / (num_spam + num_ham)
    cls_prob = pd.DataFrame({'spam_prob': prob_spam, 'ham_prob': prob_ham},
                            index=['cls_prob'])
    
    prob = pd.concat([spam_train[['spam_prob']], ham_train[['ham_prob']]],
                     axis=1)
    prob = cls_prob.append(prob)
    prob.to_csv('probabilities.csv')
    
    return prob, num_spam, num_ham

def lambda_smoothing(lambda_val, prob, num_spam, num_ham):
    """ Uses lambda smoothing in the class conditional likelihood.
    
    Args:
        lambda_val (int): Value of lambda to be used in lambda smoothing.
        prob (dataframe): Contains class conditional likelihood.
        num_spam (int): Number of mail in the training set classified as spam.
        num_ham (int): Number of mail in the training set classified as ham.

    Returns:
        prob (dataframe): Contains class conditional likelihood with
                          lambda smoothing.
        
    """
                                             
    len_vocab = len(prob[prob.index != 'cls_prob'])
    lambda_spam = lambda_val / (num_spam + lambda_val*len_vocab)
    lambda_ham = lambda_val / (num_ham + lambda_val*len_vocab)
    prob[['spam_prob', 'not_spam_prob']] = prob[['spam_prob',
                                     'not_spam_prob']].replace(0, lambda_spam)
    prob[['ham_prob', 'not_ham_prob']] = prob[['ham_prob',
                                     'not_ham_prob']].replace(0, lambda_ham)
    return prob


def test_prob(test_mail, all_mail, word_prob, spam_prob, ham_prob):
    """ Computes ham and spam probability of mail.
    
    Args:
        test_mail (datafram): Contains file path of mail to be classified.
        all_mail (dataframe): Matrix that contains presence of each vocabulary
                              word in each document
        word_prob (dataframe): Contains class conditional likelihood.
        spam_prob (float): Prior probability of spam.
        ham_prob (float): Prior probability of ham.

    Returns:
        test_mail (dataframe): Contains ham and spam likelihood of each mail.
        
    """
                                             
    word_presence = all_mail[test_mail['mail'].values[0]].astype(bool)
    spam_likelihood = sum(word_prob[word_presence].spam_prob)
    spam_likelihood += sum(word_prob[~word_presence].not_spam_prob)
    spam_likelihood += spam_prob
    spam_likelihood = math.e ** spam_likelihood
    ham_likelihood = sum(word_prob[word_presence].ham_prob)
    ham_likelihood += sum(word_prob[~word_presence].not_ham_prob)
    ham_likelihood += ham_prob
    ham_likelihood = math.e ** ham_likelihood
    test_mail['ham_prob'] = ham_likelihood / (ham_likelihood + spam_likelihood)
    test_mail['spam_prob'] = spam_likelihood / (ham_likelihood + spam_likelihood)
    return test_mail


def classifier(prob, num_spam, num_ham, lambda_val=1., num_vocab=10000,
               remove_words=[]):
    """ Classifies a mail to spam or ham.
    
    Args:
        prob (dataframe): Contains class conditional likelihood.
        num_spam (int): Number of mail in the training set classified as spam.
        num_ham (int): Number of mail in the training set classified as ham.
        lambda_val (int): Value of lambda to be used in lambda smoothing.
        num_vocab (int): Preferred number of words in the vocabulary.
        remove_words (list): List of words to be removed in the vocabulary.

    Returns:
        spam_mail_prob (dataframe): Contains predicted classification of spam 
                                    mail in test set.
        ham_mail_prob (dataframe): Contains predicted classification of ham
                                    mail in test set.

    """
    
    # Logarithmic class conditional likelihood with lambda smoothing
    prob = prob[~prob.index.isin(remove_words)][0:num_vocab+1]
    vocab_words = list(prob.index)
    if num_vocab == 200 and len(remove_words) != 0:
        with open('informative_words.txt', 'w') as text_file:
            text_file.write('\n'.join(vocab_words))
    prob['not_spam_prob'] = 1 - prob['spam_prob']
    prob['not_ham_prob'] = 1 - prob['ham_prob']
    prob = lambda_smoothing(lambda_val, prob, num_spam, num_ham)
    log_prob = np.log(prob)
    
    # Prior probability of spam and ham
    spam_prob = log_prob[log_prob.index == 'cls_prob']['spam_prob'].values[0]
    ham_prob = log_prob[log_prob.index == 'cls_prob']['ham_prob'].values[0]
    # Class conditional likelihood with lambda smoothing
    word_prob = log_prob[log_prob.index != 'cls_prob']
    
    # Predicts classification of spam mail in test set
    spam_test = pd.read_csv('spam_test.csv')
    spam_test = spam_test[spam_test.word.isin(vocab_words)]
    spam_test = spam_test.set_index('word')
    spam_mail = pd.DataFrame({'mail': spam_test.columns})
    spam_grp = spam_mail.groupby('mail')
    spam_mail_prob = spam_grp.apply(test_prob, all_mail=spam_test,
                                    word_prob=word_prob, spam_prob=spam_prob,
                                    ham_prob=ham_prob)
    spam_mail_prob['spam'] = spam_mail_prob['spam_prob'] >= spam_mail_prob['ham_prob']
    
    # Predicts classification of ham mail in test set
    ham_test = pd.read_csv('ham_test.csv')
    ham_test = ham_test[ham_test.word.isin(vocab_words)]
    ham_test = ham_test.set_index('word')
    ham_mail = pd.DataFrame({'mail': ham_test.columns})
    ham_grp = ham_mail.groupby('mail')
    ham_mail_prob = ham_grp.apply(test_prob, all_mail=ham_test,
                                    word_prob=word_prob, spam_prob=spam_prob,
                                    ham_prob=ham_prob)
    ham_mail_prob['ham'] = ham_mail_prob['ham_prob'] >= ham_mail_prob['spam_prob']

    return spam_mail_prob, ham_mail_prob


def precision_recall(spam_mail_prob, ham_mail_prob):
    """ Computes precision and recall of classifier.
    
    Args:
        spam_mail_prob (dataframe): Contains predicted classification of spam 
                                    mail in test set.
        ham_mail_prob (dataframe): Contains predicted classification of ham
                                    mail in test set.

    Returns:
        precision (float): Precision of classifier.
        recall (float): Recall of classifier.
    """

    TP = len(spam_mail_prob[spam_mail_prob.spam])
    FP = len(ham_mail_prob[~ham_mail_prob.ham])
    FN = len(spam_mail_prob[~spam_mail_prob.spam])
    precision = 1. * TP / (TP + FP)
    recall = 1. * TP / (TP + FN)
    return precision, recall


################################################################################
if __name__ == '__main__':
    start_time = datetime.now()
#    create_vocabulary()
#    word_presence_matrix()
    prob, num_spam, num_ham = probabilities()
    
    #################### USING VOCABULARY WITH 10,000 WORDS ####################
    num_vocab = 10000
    precision_list = []
    recall_list = []
    lambda_list = [2., 1., 0.5, 0.1, 0.005]
    for lambda_val in lambda_list:
        spam_mail_prob, ham_mail_prob = classifier(prob, num_spam, num_ham,
                                                   lambda_val, num_vocab)
        precision, recall = precision_recall(spam_mail_prob, ham_mail_prob)
        precision_list += [precision]
        recall_list += [recall]
    precision_list = np.array(precision_list) * 100
    recall_list = np.array(recall_list) * 100
    plt.plot(lambda_list, precision_list, label='Precision')
    plt.plot(lambda_list, recall_list, label='Recall')
    plt.legend(loc='best')
    plt.xlabel(r'Value for $\lambda$')
    plt.ylabel('Percentage')
    plt.title('Precision and Recall of Classifier \nusing Vocabulary with 10000 words')
    plt.savefig('precision and recall.png')
    plt.close()

    ###################### USING VOCABULARY WITH 200 WORDS #####################
    freq_words = list(prob[(prob.spam_prob >= 0.5) & (prob.ham_prob >= 0.5)].index)
    num_vocab = 200
    lambda_val = 0.005
    spam_mail_prob, ham_mail_prob = classifier(prob, num_spam, num_ham,
                                               lambda_val, num_vocab,
                                               freq_words)
    precision, recall = precision_recall(spam_mail_prob, ham_mail_prob)
    
    runtime = datetime.now() - start_time
    print ('runtime: ' + str(runtime))