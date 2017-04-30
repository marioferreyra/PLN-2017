"""
Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
import sys
import pickle
from docopt import docopt
from corpus.ancora import SimpleAncoraCorpusReader


def progress(msg, width=None):
    """
    Ouput the progress of something on the same line.
    """
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # Load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    PATH = "/home/mario/Escritorio/ancora-3.0.1es"
    corpus = SimpleAncoraCorpusReader(PATH, files)
    sents = list(corpus.tagged_sents())

    # Tag
    hits = 0
    total = 0

    hits_known_word = 0
    total_known_word = 0

    hits_unknown_word = 0
    total_unknown_word = 0

    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        # Tageamos la oracion con nuestro modelo
        model_tag_sent = model.tag(word_sent)

        assert len(model_tag_sent) == len(gold_tag_sent), i

        # Global score
        # ============

        # Accuracy sobre las etiquetas correctas
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits += sum(hits_sent)
        total += len(sent)
        acc_global = float(hits) / total

        length_hits_sent = len(hits_sent)

        # Accuracy sobre las palabras conocidas y palabras desconidas
        hits_known = []
        hits_unknown = []
        for index, word in enumerate(word_sent):
            value_tag = hits_sent[index]
            if model.unknown(word):
                hits_unknown.append(value_tag)
            else:
                hits_known.append(value_tag)

        hits_known_word += sum(hits_known)
        total_known_word += len(hits_known)
        acc_known_word = float(hits_known_word) / total_known_word

        hits_unknown_word += sum(hits_unknown)
        total_unknown_word += len(hits_unknown)
        acc_unknown_word = float(hits_unknown_word) / total_unknown_word

        progress("{:3.1f}% (Global {:2.2f}%) (Know {:2.2f}%) (Unknown {:2.2f}%)".format((float(i)/n)*100,
                                                                                        acc_global*100,
                                                                                        acc_known_word*100,
                                                                                        acc_unknown_word*100))

    acc_global = float(hits) / total
    acc_known_word = float(hits_known_word) / total_known_word
    acc_unknown_word = float(hits_unknown_word) / total_unknown_word

    print("")
    print("Accuracy Global: {:2.2f}%".format(acc_global * 100))
    print("Accuracy Know Words: {:2.2f}%".format(acc_known_word * 100))
    print("Accuracy Unknow Words: {:2.2f}%".format(acc_unknown_word * 100))


# Programar un script eval.py que permita evaluar un modelo de tagging. Calcular:
# * Accuracy, esto es, el porcentaje de etiquetas correctas.
# * Accuracy sobre las palabras conocidas y sobre las palabras desconocidas.
# * Matriz de confusión, como se explica en la sección 5.7.1 (Error Analysis) de Jurafsky & Martin.