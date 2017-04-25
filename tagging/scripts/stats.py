"""
Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from corpus.ancora import SimpleAncoraCorpusReader
from collections import defaultdict


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the data
    # corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    PATH = "/home/mario/Escritorio/ancora-2.0"
    corpus = SimpleAncoraCorpusReader(PATH)
    sents = list(corpus.tagged_sents())

    # Compute the statistics
    # print('sents: {}'.format(len(sents)))
    count_sents = len(sents)
    count_words = defaultdict(int)

    for sent in sents:
        pass


    # print("count sents =", count_sents)



