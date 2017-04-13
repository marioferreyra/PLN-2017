"""
Evaluate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""

import pickle
from docopt import docopt
from nltk.corpus import PlaintextCorpusReader  # Para cargar el corpus
from nltk.tokenize import RegexpTokenizer  # Tokenizador


if __name__ == '__main__':
    # Parseamos los argumentos, de las opciones
    opts = docopt(__doc__)

    # Cargamos las opciones ingresadas
    model_file = str(opts['-i'])

    # Abrimo el archivo que contiene el Modelo del lenguaje
    f = open(model_file, "rb")

    # Reconstruimos el objeto desde la representacion en cadena de bytes
    modelo = pickle.load(f)

    pattern = r'''(?ix)    # set flag to allow verbose regexps
          (?:sr\.|sra\.)
        | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        | \.\.\.            # ellipsis
        | [][.,;"'?():-_`]  # these are separate tokens; includes ]
    '''

    PATH = "/home/mario/Escritorio/Corpus"  # Ubicacion del archivo
    FILENAME = "corpus_test.txt"  # Nombre del archivo

    # Load the data
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(PATH, FILENAME, word_tokenizer=tokenizer)

    sents = corpus.sents()

    print("Perplexity =", modelo.perplexity(sents))

    # Cerramos el archivo
    f.close()
