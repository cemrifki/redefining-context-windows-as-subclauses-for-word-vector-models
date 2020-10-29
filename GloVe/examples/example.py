#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
from __future__ import print_function

import sys

from glove import Corpus
from glove import Glove

import constants

sys.path.append("..")


def main_ex(revs):
    """
    This function builds the corpus dictionary and the co-occurrence matrix.

    :return: Model is trained and saved. This function returns no value nor object.
    :rtype: None
    """
    print('Pre-processing corpus...')

    corpus_model = Corpus()
    corpus_model.fit(revs, window=constants.CONTEXT_WINDOW_SIZE)
    corpus_model.save('corpus.model')

    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    # Train the GloVe model and save it to disk.
    import time
    start_time = time.time()

    print('Training the GloVe model')

    glove = Glove(no_components=constants.EMBEDDING_SIZE, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=int(constants.GLOVE_EPOCH_NUMBER),
              no_threads=constants.GLOVE_PARALLEL_THREAD_NO, verose=True)
    glove.add_dictionary(corpus_model.dictionary)
    glove.combine_words_and_vectors()

    glove.save('glove.model')
    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
