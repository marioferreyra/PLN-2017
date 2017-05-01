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

# Para la matriz de confusion
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plotConfusionMatrix(cm,
                        classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

    # Hits Palabras conocidas
    hits_known_word = 0
    total_known_word = 0

    # Hits Palabras desconocidas
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
        total += len(sent)  # Notar que len(sent) == len(hits_sent)
        acc_global = float(hits) / total

        length_hits_sent = len(hits_sent)

        # Accuracy sobre las palabras conocidas y palabras desconidas
        hits_known = []
        hits_unknown = []
        # Analizamos cada palabra para saber si conocida o desconodida
        # El index es para saber el valor del tag de la palabra (correcto o no)
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

        progress("Porcentaje {:3.1f}% ==> \
(Global {:2.2f}%) \
(Know {:2.2f}%) \
(Unknown {:2.2f}%)".format((float(i)/n)*100,
                           acc_global*100,
                           acc_known_word*100,
                           acc_unknown_word*100))

    acc_global = float(hits) / total
    acc_known_word = float(hits_known_word) / total_known_word
    acc_unknown_word = float(hits_unknown_word) / total_unknown_word

    print("\n")
    print("Accuracy Global: {:2.2f}%".format(acc_global * 100))
    print("Accuracy Known Words: {:2.2f}%".format(acc_known_word * 100))
    print("Accuracy Unknown Words: {:2.2f}%".format(acc_unknown_word * 100))
