"""Train an n-gram model.

Usage:
  train.py -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
# from docopt import docopt
# import pickle

# from nltk.corpus import gutenberg

from nltk.corpus import PlaintextCorpusReader  # Para cargar el corpus
from nltk.tokenize import RegexpTokenizer  # Tokenizador

# from languagemodeling.ngram import NGram


if __name__ == '__main__':
    # opts = docopt(__doc__)

    pattern = r'''(?ix)    # set flag to allow verbose regexps
          (?:sr\.|sra\.)
        | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        | \.\.\.            # ellipsis
        | [][.,;"'?():-_`]  # these are separate tokens; includes ]
    '''

    PATH = "/home/mario/Escritorio"
    FILENAME = "mi_corpus.txt"

    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(PATH, FILENAME, word_tokenizer=tokenizer)

    print(corpus.sents()[:10])

    # # load the data
    # sents = gutenberg.sents('austen-emma.txt')

    # # train the model
    # n = int(opts['-n'])
    # model = NGram(n, sents)

    # # save it
    # filename = opts['-o']
    # f = open(filename, 'wb')
    # pickle.dump(model, f)
    # f.close()
