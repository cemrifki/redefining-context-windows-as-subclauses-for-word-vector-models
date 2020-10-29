#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import numpy as np
import spacy
from GloVe.glove_generator import generate_glove_vecs

from SVD_U.svd_U_generator import generate_svd_u_vecs

nlp = spacy.load("en_core_web_sm")


def generate_average_word_vecs_from_corpus(corp, word_vecs):
    """
    This function generates document embeddings in the corpus by averaging all the
    word vectors therein.

    :param corp: A list of corpus sentences that are composed of words.
    :type corp: list
    :param word_vecs: A dictionary containing words as keys and the corresponding vectors as values.
    :type word_vecs: dict
    :return: Averaged document vectors. This corresponds to all the corpus reviews.
    :rtype: np.array
    """

    mean_ = np.array(list(word_vecs.values())).mean(0)

    upd_corp = []

    for index, review in enumerate(corp):
        if index + 1 % 1000 == 0:
            print(index)
        mean_review_vec = generate_average_word_vec(review, word_vecs, mean_)
        upd_corp.append(mean_review_vec)
    return np.array(upd_corp)


def generate_average_word_vec(review, word_vecs, mean_):
    """
    This function generates a document embedding by averaging all the vectors of words
        occurring in that document.

    :param review: A text (e.g. review or e-mail) of str type.
    :type review: str
    :param word_vecs: A dictionary containing words as keys and the corresponding vectors as values.
    :type word_vecs: dict
    :param mean_: For the non-existent words, the mean vector of the whole training vectors is used.
    :type mean_: np.array or list
    :return: The averaged word vector.
    :rtype: np.array
    """

    review_vectors = np.array([word_vecs[tok] if tok in word_vecs else mean_ for tok in review])
    # review_vectors = np.array([word_vecs[tok.text] if tok.text in word_vecs else mean_ for tok in nlp(review)])
    mean_review_vec = review_vectors.mean(0)
    return mean_review_vec


def _similarity_query(word, number, word_vectors):
    """
    This function returns the most similar words to a given query.

    :param word: The target word.
    :type word: str
    :param number: The number of the most similar words to the above-given word.
    :type number: int
    :param word_vectors: word_vectors (): The set of word vectors, whereby the similarities between
            words are to be computed.
    :type word_vectors: dict
    :return: The most similar tokens to the given word.
    :rtype: list
    """

    keys, vals = list(word_vectors.keys()), list(word_vectors.values())

    word_vec = word_vectors[word]
    dst = (np.dot(vals, word_vec)
           / np.linalg.norm(vals, axis=1)
           / np.linalg.norm(word_vec))
    word_ids = np.argsort(-dst)

    return [keys[x] for x in word_ids[:number]
            if word in keys]


def generate_word_embeddings(embedding_type, revs):
    """
    This main function generates either GloVe or SVD - U embeddings.

    :param embedding_type: The type of vectors. This can be either glove or SVD_U.
    :type embedding_type: str
    :param revs: The list of reviews.
    :type revs: list or np.array
    :return: Word vectors modelled according to one of the algorithms stated above.
    :rtype: dict
    """

    import time
    start_time = time.time()

    type_dict = {"glove": generate_glove_vecs,
                 "SVD_U": generate_svd_u_vecs}

    word_vectors = type_dict[embedding_type](revs)

    elapsed_time = time.time() - start_time

    print("Word vectors have been generated. Elapsed time for it is shown below:")

    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # print(_similarity_query("more", 5, word_vectors))
    # print(_similarity_query("cat", 5, word_vectors))
    # print(_similarity_query("best", 5, word_vectors))
    # print(_similarity_query("movie", 5, word_vectors))

    return word_vectors
