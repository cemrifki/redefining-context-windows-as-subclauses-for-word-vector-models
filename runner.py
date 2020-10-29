#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın

pip3 python3 bunlar ne olacak
GloVe referansını GitHub'a koyarken mutlaka belirt.
Spam dataset'ine de refer et.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

import constants
import preprocessing
import vec_operations


def seed_everything(seed=44):
    """
    This function can be used to obtain the same results when using the same data and model.
    This can later be leveraged to perform comparison between different evaluations.

    :param seed: The seed number.
    :type seed: int
    :return: Returns nothing, just sets the seed number for different approaches.
    :rtype: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def run_svm():
    """
    This function models the SVM classifier and evaluates the performance using the test data.

    :return: Nothing. It only prints the accuracy in the end.
    :rtype: None
    """

    # seed_everything()

    clf = svm.SVC(kernel='linear', C=10)

    training_data = pd.read_csv(constants.TRAIN_FILE)
    # training_data = training_data.sample(frac=1) # This has to be adapted to the GloVe model.
    test_data = pd.read_csv(constants.TEST_FILE)

    x_train, y_train = training_data["text"], training_data["label"]
    x_test, y_test = test_data["text"], test_data["label"]

    x_train, x_test = preprocessing.preprocess_texts(x_train), preprocessing.preprocess_texts(x_test)

    context_windows = preprocessing.generate_context_windows(x_train)

    word_vectors = vec_operations.generate_word_embeddings(constants.EMBEDDING_TYPE, context_windows)

    x_train, x_test = \
        vec_operations.generate_average_word_vecs_from_corpus(x_train, word_vectors), \
        vec_operations.generate_average_word_vecs_from_corpus(x_test, word_vectors)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))


def main():
    """
    This main function handles the argument parser parameters, runs the SVM model and
        performs the evaluation on the test data.

    :return: This only trains the model and performs the evaluation.
    :rtype: None
    """

    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Fit a word embedding model.')

    parser.add_argument('--train', '-t', action='store',
                        default=10,
                        help=('Train the GloVe model with this number of epochs.'
                              'If not supplied, '
                              'We\'ll attempt to load a trained model'))

    parser.add_argument('--embedding_type', '-embedding_type', required=True, choices=["glove", "SVD_U"],
                        help="The embedding type. The two options are GloVe or SVD-U vectors.")
    parser.add_argument('--embedding_size', '-embedding_size', type=int,
                        default=300,
                        help='The length of embeddings.')
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help='Number of parallel threads to use for training')
    parser.add_argument('--stop_elim', '-st_el', action='store_true',
                        default=False,
                        help='Stop word elimination.')
    parser.add_argument('--subclause', '-subcl', action='store_true',
                        default=False,
                        help='Employ subclauses or sliding windows.')
    parser.add_argument('--symmetric', '-sym', action='store_true',
                        default=True,
                        help=('If sliding windows are used, set the '
                              'symmetric windows paradigm. If False, '
                              'use right context windows instead'))
    parser.add_argument('--context_window_size', '-cont_size', type=int,
                        default=10,
                        help='If sliding windows are used, set the size of context windows.')
    parser.add_argument('--include_ccomp', '-incl_ccomp', action='store_true',
                        default=True,
                        help=('If subclauses are used, determine whether clausal components are '
                              'taken into account when generating subsentences as well.'))

    parser.add_argument('--training_dataset', default="input/Sentiment_training.csv",
                        help='Sentiment or Spam')
    parser.add_argument('--test_dataset', default="input/Sentiment_test.csv", help='Sentiment or Spam')

    args = parser.parse_args()

    constants.EMBEDDING_SIZE = args.embedding_size
    constants.EMBEDDING_TYPE = args.embedding_type
    constants.STOPWORD_ELIM = args.stop_elim
    constants.SUBCLAUSE = args.subclause
    constants.SYMMETRIC = args.symmetric
    constants.CONTEXT_WINDOW_SIZE = args.context_window_size
    constants.INCLUDE_CCOMP = args.include_ccomp
    constants.TRAIN_FILE = args.training_dataset
    constants.TEST_FILE = args.test_dataset

    constants.GLOVE_PARALLEL_THREAD_NO = args.parallelism
    constants.GLOVE_EPOCH_NUMBER = args.train

    run_svm()


if __name__ == "__main__":
    main()
    # Some examples to run the code are given below.
    # python3 runner.py --subclause --embedding_type SVD_U --embedding_size 300 --training_dataset input/Sentiment_training.csv --test_dataset input/Sentiment_test.csv
    # python3 runner.py --symmetric --embedding_type SVD_U --embedding_size 300 --training_dataset input/Spam_training.csv --test_dataset input/Spam_test.csv
    # python3 runner.py --subclause --include_ccomp --embedding_type glove --embedding_size 100 --stop_elim --training_dataset input/Sentiment_training.csv --test_dataset input/Sentiment_test.csv 
