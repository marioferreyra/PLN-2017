"""
Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""

import pickle
from docopt import docopt
from nltk.corpus import PlaintextCorpusReader  # Para cargar el corpus
from nltk.tokenize import RegexpTokenizer  # Tokenizador
from languagemodeling.ngram import NGram, AddOneNGram


if __name__ == '__main__':
    # Parseamos los argumentos, de las opciones
    opts = docopt(__doc__)

    pattern = r'''(?ix)    # set flag to allow verbose regexps
          (?:sr\.|sra\.)
        | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        | \.\.\.            # ellipsis
        | [][.,;"'?():-_`]  # these are separate tokens; includes ]
    '''

    PATH = "/home/mario/Escritorio/Corpus"  # Ubicacion del archivo
    FILENAME = "corpus_train.txt"  # Nombre del archivo

    # Load the data
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(PATH, FILENAME, word_tokenizer=tokenizer)

    sents = corpus.sents()

    # Train the model
    n = int(opts['-n'])
    m = str(opts['-m'])

    # Podemos usar los N-Gramas clasicos o N-Gramas con el suavizado AddOne
    if m == "addone":
        model = AddOneNGram(n, sents)
    else:
        model = NGram(n, sents)

    # Save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
