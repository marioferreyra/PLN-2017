"""
Train a sequence tagger.

Usage:
  train.py [-m <model>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: base]:
                  base: Baseline
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger


models = {
    'base': BaselineTagger,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    PATH = "/home/mario/Escritorio/ancora-3.0.1es"
    corpus = SimpleAncoraCorpusReader(PATH, files)
    sents = list(corpus.tagged_sents())

    # Train the model
    m = str(opts['-m'])
    print("##### Baseline Tagger #####")
    model = models[m](sents)

    # Save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
