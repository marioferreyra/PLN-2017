"""
Train a parser.

Usage:
  train.py [-m <model>] [-n <order>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: flat]:
                  flat: Flat trees
                  rbranch: Right branching trees
                  lbranch: Left branching trees
                  upcfg: Unlexicalized Probabilistic Context-Free Grammar trees
  -n <order>    Order of Horizontal Markovization
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.baselines import Flat, LBranch, RBranch
from parsing.upcfg import UPCFG


models = {
    'flat': Flat,
    'rbranch': RBranch,
    'lbranch': LBranch,
    'upcfg': UPCFG
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading corpus ...')
    PATH = "/home/mario/Escritorio/ancora-3.0.1es"
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader(PATH, files)

    print('Training model ...')
    # x = list(corpus.parsed_sents())[:10]
    m = opts['-m']  # Modelo Elegido
    n = opts['-n']  # Orden Markovizacion Horizontal
    if (n is not None) and (m == "upcfg"):
        model = models[opts['-m']](corpus.parsed_sents(), horzMarkov=int(n))
    else:
        model = models[opts['-m']](corpus.parsed_sents())
    # model = models[opts['-m']](corpus.parsed_sents())
    # x = corpus.parsed_sents()
    # model = models[opts['-m']](x)

    print('Saving ...')
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
