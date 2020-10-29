#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""


from scipy.sparse.linalg import svds

import constants
from SVD_U.PMI_matrix import prepare_pmi


class CorpusSvdU:
    """
    This class implements the SVD - U model using the training data.
    """
    def __init__(self):
        """
        Default constructor.
        """
        pass

    def get_corpus_svd_U_vecs(self, revs):
        """
        This function generates a dict whose keys are words and values are the corresponding
        "U" vectors.

        :param revs: All corpus reviews.
        :type revs: list
        :return: The above-mentioned dict.
        :rtype: dict
        """
        PMI_dict_keys, PMI_matr = prepare_pmi(revs)
        
        return get_svd_U(PMI_dict_keys, PMI_matr)


def get_svd_U(words, matr):
    """
    This function gets the "U" vectors of the words.

    :param words: The set of words.
    :type words: Set or list
    :param matr: The matrix containing the PMI information for each word entry.
    :type matr: list or np.array
    :return: Words and their vectors.
    :rtype: dict
    """
    dim_size = constants.EMBEDDING_SIZE
    if dim_size > len(matr[0]):
        raise Exception("The size of embedding should have been lower than the number of words in the corpus.")
        
    U, _, _ = svds(matr, k=dim_size)
    
    res_dict = dict(zip(words, U))
    
    return res_dict


def generate_svd_u_vecs(revs):
    """
    This function generates the above-mentioned "U" vectors for corpus words.

    :param revs: All corpus reviews.
    :type revs: list
    :return: The dictionary whose keys are words and values are the corresponding "U" vectors.
    :rtype: dict
    """
    word_vectors = CorpusSvdU().get_corpus_svd_U_vecs(revs)

    return word_vectors


if __name__ == "__main__":
    pass
