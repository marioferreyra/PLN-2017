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
                  inter: N-grams with interpolation smoothing.
                  backoff: N-grams with back-off smoothing (with discounting).
  -o <file>     Output model file.
  -h --help     Show this screen.
"""

import pickle
from docopt import docopt
from nltk.corpus import PlaintextCorpusReader  # Para cargar el corpus
from nltk.tokenize import RegexpTokenizer  # Tokenizador
from languagemodeling.ngram import NGram, AddOneNGram
from languagemodeling.ngram import InterpolatedNGram, BackOffNGram


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

    PATH = "./../../Corpus_Language_Modeling"  # Ubicacion del archivo
    FILENAME = "corpus_train.txt"  # Nombre del archivo

    # Load the data
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(PATH, FILENAME, word_tokenizer=tokenizer)

    sents = corpus.sents()

    # Train the model
    n = int(opts['-n'])
    m = str(opts['-m'])

    # Podemos usar los N-Gramas clasicos, N-Gramas suavizado AddOne,
    # N-Gramas suavizado por Interpolacion o N-Gramas suavizado por Back-Off
    if m == "addone":
        print("##### Suavizado AddOne #####")
        model = AddOneNGram(n, sents)
    elif m == "inter":
        print("##### Suavizado por Interpolacion #####")
        model = InterpolatedNGram(n, sents)
    elif m == "backoff":
        print("##### Suavizado por Back-Off #####")
        model = BackOffNGram(n, sents)
    else:
        print("##### N-Gramas clasicos #####")
        model = NGram(n, sents)

    # Save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
