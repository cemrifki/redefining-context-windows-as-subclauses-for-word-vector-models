#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""

import os
import pickle
import subprocess
import sys

from GloVe.examples import example

sys.path.insert(0, '..')

def generate_glove_vecs(revs):
    """
    This function generates GloVe vectors based on the training data. This function
        can be more optimized in future.

    :return: A dictionary containing words as keys and their GloVe vectors as the corresponding values.
    :rtype: dict
    """

    os.chdir("GloVe")

    subprocess.call(['python3', 'setup.py', 'cythonize'])
    os.system("pip3 install -e .")
    os.chdir("..")

    example.main_ex(revs)
    word_vectors = pickle.load(open("glove.model", "rb"))["words_and_vectors"]

    return word_vectors
