"""
Constant parameters to be leveraged across the program.

"""
import os

EMBEDDING_SIZE = 50

EMBEDDING_TYPE = "glove"

STOPWORD_ELIM = False

TRAIN_FILE = os.path.join("input", "Sentiment_training.csv")
TEST_FILE = os.path.join("input", "Sentiment_test.csv")

# Variable defining our approach, where subclauses are defined as context windows.
# This needs to be True to be activated.
SUBCLAUSE = False

# The below variabled is used only for the baseline approach. If it is set as false,
# only the right context windows of target words are used.
SYMMETRIC = False

CONTEXT_WINDOW_SIZE = 10

# If the below variable is set as True, we generate more subclauses,
# when we encounter a clausal component relation in the dependency tree.
INCLUDE_CCOMP = True

# The below two variables are only relevant to the GloVe model.
GLOVE_EPOCH_NUMBER = 10
GLOVE_PARALLEL_THREAD_NO = 1

MODEL_FILE_NAME = "corpus_subclauses.sav"
