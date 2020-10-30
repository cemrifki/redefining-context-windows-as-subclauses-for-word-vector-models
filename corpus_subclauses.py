#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""

import re
from collections import defaultdict

import spacy

import constants
import preprocessing

# Punctuation marks across the module can be handled more efficiently and consistently in future.
p = re.compile(r"([.?!])[\"\']*$")

nlp = spacy.load("en_core_web_sm")

conj_and_punc_list = ["and", "or", "but", "however", "also", "?", "!", ".", ",", ":", ";", ]


class CorpusSubclauses:
    """
    This class is used to generate subclauses from texts.
    """
    def __init__(self):
        pass

    def final_punc_mark(self, s):
        """
        This method is used to detect the final pucntuation mark at the end of the text.

        :param s: Text string.
        :type s: str
        :return: Punctuation mark at the end of the text.
        :rtype: str
        """

        res = re.search(p, s)
        final_punc = res.group(1) if res else "."
        return final_punc

    def convert_to_subclause_corpus(self, sentences):
        """
        This method is used to convert the reviews of the whole corpus to their corresponding
            subclauses.

        :param sentences: The list of corpus texts.
        :type sentences: list
        :return: All the subclauses in the corpus.
        :rtype: list
        """

        all_subclauses = []
        for sentence in sentences:
            str_sentence_arr = [re.sub(r"\"", r"", word.lower()) for word in sentence]
            str_sentence = " ".join(str_sentence_arr)
            final_punc_mark = self.final_punc_mark(str_sentence)
    
            str_sentence = re.sub(r"[ ]+", r" ", str_sentence)
            rev_subclauses = self.get_all_subclauses_of_sent(str_sentence)
            # The below block is used to add the final punctuation mark at the end of the sentence to the end of
            # every subclause as well to provide consistency.
            for ind, rev_subclause in enumerate(rev_subclauses):
                adapted_subcl = self.remove_trailing_puncs(" ".join(rev_subclause))
                # The final punctuation mark of the last sentence is appended to the end of all
                # sentences. This could be updated and enhanced.
                adapted_subcl += " " + final_punc_mark

                updated_sentence = adapted_subcl.strip().split()

                all_subclauses.append(updated_sentence)
        return all_subclauses

    def convert_sentence_to_subclauses(self, sentence):
        """
        The helper method that converts a sentence as a whole in string format to its corresponding subclauses.
        This actually is not used in this project, but can be employed by anyone who wants
            to break a sentence in string format into its subclauses as list.

        :param sentence: An input string: sentence to be broken down into its subclauses.
        :type sentence: str
        :return: The result subclauses from this sentence.
        :rtype: list
        """

        tokenized_text = preprocessing.normalize_tokenize(sentence)
        str_sentence_arr = [tok.text for tok in nlp(tokenized_text)]

        str_sentence = " ".join(str_sentence_arr)
        final_punc_mark = self.final_punc_mark(str_sentence)

        str_sentence = re.sub(r"[ ]+", r" ", str_sentence)
        rev_subclauses = self.get_all_subclauses_of_sent(str_sentence)

        subclauses_with_puncs = []

        for ind, rev_subclause in enumerate(rev_subclauses):
            adapted_subcl = self.remove_trailing_puncs(" ".join(rev_subclause))
            # The final punctuation mark of the last sentence is appended to the end of all
            # sentences. This could be updated and enhanced.
            adapted_subcl += " " + final_punc_mark

            updated_sentence = adapted_subcl.strip().split()
            subclauses_with_puncs.append(updated_sentence)
        return subclauses_with_puncs

    def get_children_recurs(self, tok, lev):
        """
        This method finds the children nodes of a target word recursively, all of which
            on a whole constitute a subclause.
        If the dependency relations "ccomp" or "conj" are encountered, it marks the existence
            of a new subclause and therefore the recursively scanning mechanism is stopped.
            However, this rule set can be redefined by updatin the if statament below, which is
            if (dep == "conj" or dep == "ccomp").

        :param tok: The target token whose children in the subclause are to be generated.
        :type tok: spacy.tokens.token.Token
        :param lev: The height (level) of the recursive tree.
        :type lev: int
        :return: All the kids of a token in a recursive subclause tree.
        :rtype: set
        """

        all_children = set([])
        if lev == 0:
            all_children.add(tok.text + "-" + str(tok.i))
        lev += 1
        for kid in tok.children:
            tag = kid.tag_.lower()
            dep = kid.dep_.lower()
            # The below if statement segments the text into subclauses if the parser
            # encounters a "ccomp" relationship as well in addition to the "conj" relationship.
            if constants.INCLUDE_CCOMP:
                if (dep == "conj" or dep == "ccomp") and (tag == "vbd" or tag == "vbz" or tag == "vbg"):
                    continue
            # The below statement partitions the text only with respect to the "conj" relationship.
            else:
                if (dep == "conj") and (tag == "vbd" or tag == "vbz" or tag == "vbg"):
                    continue
            all_children.add(kid.text + "-" + str(kid.i))
            all_children |= self.get_children_recurs(kid, lev)
        return all_children

    def get_all_deps(self, sentence):
        """
        This helper method generates all the tokens which are connected to each other
        via a dependency relationship.

        :param sentence: The input text (e.g. sentence).
        :type sentence: str
        :return: All the words connected to each other via dependency relationships in the
            same subclasue.
        :rtype: list
        """

        doc = nlp(sentence)
        res = [self.get_children_recurs(token, 0) for token in doc]
        return res

    def get_all_subclauses_of_sent(self, sentence):
        """
        This helper method generates the set of all subclauses from a given sentence.

        :param sentence: The input text (e.g. sentence).
        :type sentence: str
        :return: All the subclauses generated from "sentence".
        :rtype: list
        """

        sent = sentence.strip()
        final_punc = sent[-1]
        if not re.match(r"[?.!]+$", final_punc):
            final_punc = "."
        subclauses = defaultdict(set)
        all_deps = self.get_all_deps(sent)
        all_toks = set({})
        for dep in all_deps:
            for tok in dep:
                all_toks.add(tok)
                if len(subclauses[tok]) < len(dep):
                    subclauses[tok] = dep
        final_subclauses = set([tuple(subclauses[tok]) for tok in all_toks])

        final_subclauses_tmp = set({})
        for ind, subcl in enumerate(final_subclauses):
            subcl = list(subcl)
            subcl.sort(key=lambda token: int(token[token.rindex("-") + 1:]))
            final_subclauses_tmp.add(tuple(subcl))
        final_subclauses_upd = []

        for subcl in final_subclauses_tmp:
            subcl = list(subcl)
            subcl = self.remove_ind(subcl)
            if subcl[-1] not in ".?!;:":
                subcl = subcl + [final_punc]
            subcl = self.remove_trailing_conjs_and_puncs(subcl, conj_and_punc_list)
            final_subclauses_upd.append(subcl)
        return final_subclauses_upd

    def remove_ind(self, s):
        """
        This is a helper method, which removes the hyphen and index information
        from the end of a token. This is a supplementary function.

        :param s: A list of words (i.e. sentence or a whole review).
        :type s: list
        :return: Words stripped of their hyphen and rudimentary suffixes.
        :rtype: list
        """

        return [re.sub(re.compile(r"(.*)-[0-9]+"), r"\1", tok) for tok in s]

    def remove_trailing_puncs(self, s):
        """
        A helper method that removes the trailing punctuation marks.

        :param s: A text input.
        :type s: str
        :return: The text stripped off its unnecessary punctuation marks.
        :rtype: str
        """

        return re.sub(r"[,;:.?!\"]+$", r"", s)

    def remove_trailing_conjs_and_puncs(self, s, conjunction_list):
        """
        This method removes the trailing conjunctions and punctuation marks that are specified
        in the variable conj_and_punc_list.
        This is useful, since a subclause cannot in general start with a punctuation mark or, say "and".


        :param s: A list of tokens (e.g. a sentence).
        :type s: list
        :param conjunction_list: The list of conjunctions and punctuation marks.
        :type conjunction_list: list
        :return: The sentence or subclause stripped off its unnecessary conjunctions or
            punctuation marks in the beginning or end.
        :rtype: list
        """

        if len(s) <= 1:
            return s
        while True:
            neither_conj = 0
            if s[0] in conjunction_list:
                s = s[1:]
                neither_conj += 1
            if s[-1] in conjunction_list:
                s = s[:-1]
                neither_conj += 2
            if neither_conj == 0 or len(s) <= 1:
                break
        return s


if __name__ == "__main__":
    cs = CorpusSubclauses()

    # The below is an example that converts a sentence into the subclauses thereof.
    print(cs.convert_sentence_to_subclauses("The service was awesome, and the food was incredible!"))

    # The below are examples that convert a corpus of sentences to their counterpart subclauses.
    print(cs.convert_to_subclause_corpus(
        [["The vibe is relaxed and cozy, the service was great, and the ambiance was good!"]]))
    print(cs.convert_to_subclause_corpus([["I loved that too much!"]]))
    print(cs.convert_to_subclause_corpus([["And she said: \"God, I loved this movie so much!\""]]))
    print(cs.convert_to_subclause_corpus([["I just moved here. However, I disliked this city!"]]))
    print(cs.convert_to_subclause_corpus([["If you are not going to play this song at my funeral, "
                                           "then I won't be attending!"]]))
    print(cs.convert_to_subclause_corpus([["The ambiance here is warm, however the prices are high."]]))
