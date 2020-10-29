#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module is created to handle preprocessing operations.
@author: Cem Rıfkı Aydın
"""

import re
from collections import Counter

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import constants
from corpus_subclauses import CorpusSubclauses

nlp = spacy.load("en_core_web_sm")
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')


def normalize_tokenize(string):
    """
    This is a basic, generic tokenizer.

    :param string: Text to be tokenized.
    :type string: str
    :return: String in a tokenized form that is to be split later by other helper methods.
    :rtype: str
    """
    string = string.lower()
    # HTML tags are removed by the below line of code.
    string = tag_re.sub(' ', string)
    # This is added to reduce the same characters appearing consecutively more than twice
    # to the same two chars only.
    string = re.sub(r"(.)(\1)\1{2,}", r"\1\1\1", string)
    # Added, since mentions (e.g. @trump) do not contribute to sentiment.
    string = re.sub(r"@[a-zA-Z0-9()#,!?:=;\-\\'`./]+", r"", string)
    # Some extra chars are added to be taken into account.
    string = re.sub(r"[^A-Za-z0-9()#,!?:=;\-\\'`./]", " ", string)

    # Numeric forms and emoticons, such as 22:30, are not disrupted.
    string = re.sub(r"([^\d)(])([,.:;]+)([^\d()]|$)", r"\1 \2 \3", string)
    # The punctuation marks "?" and "!" can be indicative of expressing sentiment. These
    # are therefore not removed.
    string = re.sub(r"([!?]+)", r" \1 ", string)

    # The below four regex commands are implemented to put blank spaces before or after
    # parens without disrupting emoticons.

    string = re.sub(r"\(([A-Za-z0-9,!?\-\\'`])", r"( \1", string)
    string = re.sub(r"([A-Za-z0-9,!?\-\\'`])\(", r"\1 (", string)

    string = re.sub(r"([A-Za-z0-9,!?\-\\'`])\)", r"\1 )", string)
    string = re.sub(r"\)([A-Za-z0-9,!?\-\\'`])", r") \1", string)

    # "(?!)" and similar forms that likely indicate sarcasm are kept.
    string = re.sub(r"(\() +([?!]+) +(\))", r"\1\2\3", string)
    # Useless parens are removed.
    string = re.sub(r"(^|[ ])+([()]+)([ ]+|$)", r" ", string)
    # Other useless punctuations are also eliminated.
    string = re.sub(r"(^|[ ])+([.;,]+)([ ]+|$)", r" ", string)

    # Emoticons ":s" and ":D".
    string = re.sub(r"((\s|^)[:]+)[ ]+([dDsSpP]+(\s|$))", r"\1\3", string)
    # Emoticon handling.
    string = re.sub(r"([:;.]+)([()dDsSpP]+)", r" \1\2 ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string


def generate_context_windows(preprocessed_texts):
    """
    This function generates context windows, which can be defined as either subclauses or sliding windows.
        This also removes noise words that are too infrequently occurring words.

    :param preprocessed_texts: Input that are previously preprocessed and tokenized.
    :type preprocessed_texts: List of lists
    :return: Subclauses or sliding windows as context windows.
    :rtype: List of lists
    """
    if constants.SUBCLAUSE:
        cs = CorpusSubclauses()
        preprocessed_texts = cs.convert_to_subclause_corpus(preprocessed_texts)
    print("The texts have been preprocessed.")
    preprocessed_texts = remove_noise(preprocessed_texts)
    return preprocessed_texts


def preprocess_texts(texts):
    """
    This function performs preprocessing using spaCy's tokenizer in addition to my basic tokenizer.

    :param texts: Input texts (corpus).
    :type texts: List of strings
    :return: Preprocessed texts (reviews, mails, etc.).
    :rtype: List of lists
    """
    updated_texts = []
    for text in texts:
        text = normalize_tokenize(text)
        updated_texts.append([tok.text for tok in nlp(text)])
    return updated_texts


def get_all_context_words(all_revs, context_window_size=constants.CONTEXT_WINDOW_SIZE):
    """
    This function generates a dictionary whose keys are words and values are the corresponding
        words occurring in their context windows.

    :param all_revs: All corpus reviews.
    :type all_revs: list or np.array
    :param context_window_size: The size of context windows.
    :type context_window_size: int
    :return: A dictionary holding the information stated above.
    :rtype: dict
    """

    contexts_of_words = {}  # dict(Counter(int))
    if constants.SYMMETRIC:
        context_window_size = int(context_window_size / 2)
    for rev in all_revs:
        for i in range(0, len(rev)):
            target_word = rev[i]

            if constants.SYMMETRIC:

                start_index = max(i - context_window_size, 0)
                end_index = min(i + context_window_size + 1, len(rev))

            else:
                start_index = i
                end_index = min(i + context_window_size + 1, len(rev))

            left_hand_side_els, right_hand_side_els = [], []
            if i > 0:
                left_hand_side_els = rev[start_index:i]
            if i < len(rev) - 1:
                right_hand_side_els = rev[i + 1:end_index]

            word_context_words = set(left_hand_side_els + right_hand_side_els)
            if constants.STOPWORD_ELIM:
                word_context_words = set([word for word in word_context_words if word not in STOP_WORDS])
            if target_word in word_context_words:
                word_context_words.remove(target_word)
            word_context_words_curr_dct = Counter(word_context_words)

            if target_word not in contexts_of_words:
                contexts_of_words[target_word] = word_context_words_curr_dct
            else:
                contexts_of_words[target_word] = contexts_of_words[target_word] + word_context_words_curr_dct

    return contexts_of_words


def remove_noise(revs):
    """
    This optional helper method removes noise words that are too infrequent in the corpus.
    These words are considered to not be discriminative for the sentiment classification task.

    :param revs: All corpus reviews.
    :type revs: list
    :return: Corpus reviews stripped off noise words.
    :rtype: list
    """

    noise_threshold = (1 / 1000) * len(revs)  # Could also be defined as cnt=constants.NOISE_THRESHOLD_VAL.
    word_cnt = Counter()
    for rev in revs:
        rev = set(rev)
        for word in rev:
            word_cnt[word] += 1

    revs_without_noise = []
    for rev in revs:
        revs_without_noise.append([word for word in rev if word_cnt[word] > noise_threshold])
    return revs_without_noise
