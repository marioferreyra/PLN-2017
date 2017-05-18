"""
Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <n>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: base]:
                  base: Baseline
                  mlhmm: Maximum Likehood Hidden Markov Model
                  memm: Maximum Entropy Hidden Markov Model
  -n <n>        Order of the model MLHMM and MEMM.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM
from tagging.memm import MEMM


models = {
    "base": BaselineTagger,
    "mlhmm": MLHMM,
    "memm": MEMM
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
    if m in {"mlhmm", "memm"}:
        n = int(opts['-n'])

    if m == "mlhmm":
        print("##### MLHMM #####")
        model = models[m](n, sents)
    elif m == "memm":
        print("##### MEMM #####")
        model = models[m](n, sents)
    else:
        print("##### Baseline Tagger #####")
        model = models[m](sents)

    # Save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
