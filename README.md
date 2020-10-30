# Redefining Context Windows as Subclauses for Word Vector Models

This approach attempts to model word vector models by defining subclauses as context windows. The way the subclauses are generated in this study is novel. In this study, we modelled both GloVe and SVD - U methods. The GloVe method we relied on is mostly based on a repository we have found on the Web. We updated it a bit to adapt it to our approach. On the other hand, the codes for the SVD - U technique and other modules are mostly written by us.

## Requirements
- Python 3.7 or a newer version
- Cython
- numpy
- pandas
- pytest
- scikit_learn
- scipy
- setuptools
- spacy

 The code can work with Python 3.7 or a newer version. In this project, `python3` and `pip3` commands are utilized for the GloVe model. We leveraged two datasets, which are sentiment and spam corpora and which can be found in the input folder.
## Execution

Execute the file `runner.py` to train word and document embeddings and evaluate the model on the test dataset.
The following are the command-line arguments:

- `--subclause`: set whether or not you define context windows as subclauses
- `--embedding_type`: set embedding type, which can be `glove` or `SVD_U`
- `--embedding_size`: set embedding size
- `--train`: set epoch number for the GloVe model
- `--stop_elim`: set whether or not you perform stop word elimination
- `--context_window_size`: set length of context windows
- `--symmetric`: set whether you use symmetric context windows when choosing the sliding windows technique
- `--include_ccomp`: set whether you take into account the "ccomp" dependency relationship as well when generating subclauses
- `--training_dataset`: set training file path (e.g. "input/Sentiment_training.csv")
- `--test_dataset`: set test file path (e.g. "input/Sentiment_test.csv")

#### Setup with virtual environment (Python 3):
-  python3 -m venv my_venv
-  source my_venv/bin/activate

Install the requirements:
-  pip3 install -r requirements.txt

If everything works well, you can run the example usage given below.

### Example Usage:
- The following guide shows an example usage of the model in generating word and document vectors using the GloVe or SVD - U model, and performing the evaluation.
- Instructions
      
      1. Change directory to the location of the source code
      2. Run the instructions in "Setup with virtual environment (Python 3)"
      3. Run the runner.py file with chosen command parameters. Some examples are given below

Examples:
```
python3 runner.py --subclause --embedding_type SVD_U --embedding_size 300 --training_dataset input/Sentiment_training.csv --test_dataset input/Sentiment_test.csv
python3 runner.py --symmetric --embedding_type SVD_U --embedding_size 300 --training_dataset input/Spam_training.csv --test_dataset input/Spam_test.csv
python3 runner.py --subclause --include_ccomp --embedding_type glove --embedding_size 100 --stop_elim --training_dataset input/Sentiment_training.csv --test_dataset input/Sentiment_test.csv 
```
## Citation
If you find this code useful, please cite the following in your work:
```
@phdthesis{cra:20,
  author       = {Cem Rifki Aydin}, 
  title        = {Developing a Comprehensive Framework for Sentiment Analysis in Turkish},
  school       = {Bogazici University},
  year         = 2020
}
```
## Credits
Codes were written by Cem Rıfkı Aydın
