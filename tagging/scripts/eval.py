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
from sklearn.metrics import confusion_matrix  # Para la matriz de confusion
from collections import Counter


def printConfusionMatrix(labels, confusion_matrix):
    length_labels = len(labels)
    initial_delimiter = True
    for label in labels:
        if initial_delimiter:
            print("| {:^7} |".format(""), end=" ")
            initial_delimiter = False
        print("{:^7}".format(label), end=" | ")

    initial_delimiter = True
    for i in range(length_labels+1):
        if initial_delimiter:
            print("\n|", end="")
            initial_delimiter = False
        print("{}".format(":-------:"), end="|")

    print("")
    for row in range(length_labels):  # Filas
        initial_delimiter = True
        for column in range(length_labels):  # Columnas
            value = round(my_confusion_matrix[row][column], 2)
            if initial_delimiter:
                print("| {:^7} |".format(labels[row]), end=" ")
                initial_delimiter = False
            if value == 0.0:
                print("{:^7}".format("-"), end=" | ")
            else:
                print("{:^7}".format(value), end=" | ")
        print("")
    print("")


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

    # Para Matriz de Confusion
    tags_gold = []  # Tags correctos
    tags_models = []  # Tags modelados

    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        # Tageamos la oracion con nuestro modelo
        model_tag_sent = model.tag(word_sent)

        tags_gold += list(gold_tag_sent)
        tags_models += model_tag_sent

        print(len(model_tag_sent), len(gold_tag_sent))
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

#         progress("Porcentaje {:3.1f}% ==> \
# (Global {:2.2f}%) \
# (Know {:2.2f}%) \
# (Unknown {:2.2f}%)".format((float(i)/n)*100,
#                            acc_global*100,
#                            acc_known_word*100,
#                            acc_unknown_word*100))

    acc_global = float(hits) / total
    acc_known_word = float(hits_known_word) / total_known_word
    acc_unknown_word = float(hits_unknown_word) / total_unknown_word

    print("\n")
    print("Accuracy Global: {:2.2f}%".format(acc_global * 100))
    print("Accuracy Known Words: {:2.2f}%".format(acc_known_word * 100))
    print("Accuracy Unknown Words: {:2.2f}%".format(acc_unknown_word * 100))

    # Computamos Matriz de Confusion
    print("\nMatriz de confusion")
    print("===================")
    counter_tags = Counter(tags_gold)
    # Obtenemos los 10 tags mas frecuentes
    labels = [t for t, _ in counter_tags.most_common(10)]
    my_confusion_matrix = confusion_matrix(tags_gold, tags_models, labels)
    my_confusion_matrix = (my_confusion_matrix/total)*100

    printConfusionMatrix(labels, my_confusion_matrix)
