"""
Evaluate a parser.

Usage:
  eval.py -i <file> [-m <m>] [-n <n>]
  eval.py -h | --help

Options:
  -i <file>     Parsing model file.
  -m <m>        Parse only sentences of length <= <m>.
  -n <n>        Parse only <n> sentences (useful for profiling).
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.util import spans


def progress(msg, width=None):
    """
    Ouput the progress of something on the same line.
    """
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


def precision(hits, total_model):
    """
    Calcula la Precision.
    """
    return (float(hits) / total_model) * 100


def recall(hits, total_gold):
    """
    Calcula la Recall.
    """
    return (float(hits) / total_gold) * 100


def f1(precision, recall):
    """
    Calcula F1.
    """
    return (2*precision*recall) / (precision + recall)


if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading model ...')
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    print('Loading corpus ...')
    PATH = "/home/mario/Escritorio/ancora-3.0.1es"
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader(PATH, files)
    parsed_sents = list(corpus.parsed_sents())

    # Opcion para seleccionar las primeras n oraciones
    n = opts["-n"]
    if n is not None:
        n = int(n)
        parsed_sents = parsed_sents[:n]

    # Opcion para seleccionar las oraciones de largo <= m
    m = opts["-m"]
    if m is not None:
        m = int(m)
        new_parsed_sents = []
        for parsed_sent in parsed_sents:
            if len(parsed_sent.leaves()) <= m:
                new_parsed_sents += [parsed_sent]

        parsed_sents = new_parsed_sents

    print('Parsing ...')
    labeled_hits = 0
    unlabeled_hits = 0
    total_gold = 0
    total_model = 0

    n = len(parsed_sents)

    format_str = '{:3.1f}% ({}/{}) (Labeled_P={:2.2f}%, Labeled_R={:2.2f}%,\
Labeled_F1={:2.2f}%) (Unlabeled_P={:2.2f}%, Unlabeled_R={:2.2f}%,\
Unlabeled_F1={:2.2f}%)'
    progress(format_str.format(0.0, 0, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    for i, gold_parsed_sent in enumerate(parsed_sents):
        tagged_sent = gold_parsed_sent.pos()

        # Parse
        model_parsed_sent = model.parse(tagged_sent)

        # Compute labeled scores
        labeled_gold_spans = spans(gold_parsed_sent, unary=False)
        labeled_model_spans = spans(model_parsed_sent, unary=False)

        # Compute unlabeled scores
        unlabeled_gold_spans = set()
        for element in labeled_gold_spans:
            unlabeled_gold_spans.add(element[1:])

        unlabeled_model_spans = set()
        for element in labeled_model_spans:
            unlabeled_model_spans.add(element[1:])

        # Compute hits
        labeled_hits += len(labeled_gold_spans & labeled_model_spans)
        unlabeled_hits += len(unlabeled_gold_spans & unlabeled_model_spans)

        # Compute total
        total_gold += len(labeled_gold_spans)
        total_model += len(labeled_model_spans)

        # Compute labeled partial results
        labeled_prec = precision(labeled_hits, total_model)
        labeled_rec = precision(labeled_hits, total_gold)
        labeled_f1 = f1(labeled_prec, labeled_rec)

        # Compute labeled partial results
        unlabeled_prec = precision(unlabeled_hits, total_model)
        unlabeled_rec = precision(unlabeled_hits, total_gold)
        unlabeled_f1 = f1(unlabeled_prec, unlabeled_rec)

        progress(format_str.format(float(i+1)*100 / n,
                 i+1,
                 n,
                 labeled_prec,
                 labeled_rec,
                 labeled_f1,
                 unlabeled_prec,
                 unlabeled_rec,
                 unlabeled_f1))

    print("")
    print("Parsed {} sentences".format(n))
    print("")
    print("##### Labeled")
    print("* Precision = {:2.2f}%".format(labeled_prec))
    print("* Recall = {:2.2f}%".format(labeled_rec))
    print("* F1 = {:2.2f}%".format(labeled_f1))
    print("")
    print("##### Unlabeled")
    print("* Precision = {:2.2f}%".format(unlabeled_prec))
    print("* Recall = {:2.2f}%".format(unlabeled_rec))
    print("* F1 = {:2.2f}%".format(unlabeled_f1))
