"""
Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from corpus.ancora import SimpleAncoraCorpusReader
from collections import defaultdict, OrderedDict


def loadAncoraCorpus():
    # Load the data
    # corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    PATH = "/home/mario/Escritorio/ancora-2.0"
    corpus = SimpleAncoraCorpusReader(PATH)
    sents = list(corpus.tagged_sents())

    return sents


if __name__ == '__main__':
    opts = docopt(__doc__)

    sents = loadAncoraCorpus()

    # Compute the statistics
    count_sents = len(sents) # Cantidad de oraciones
    dict_words = defaultdict(int) # Cantidad de palabras
    dict_tags = defaultdict(int) # Cantidad de tags

    # tag : {word : cantidad}
    dict_tag_word_count = defaultdict(lambda : defaultdict(int)) # Capaz podamos simplicar algo con esta estructura

    for sent in sents:
        # sent es una lista de pares: (palabra, tag)
        for word, tag in sent:
            dict_words[word.lower()] += 1 # Para no diferenciar entre "Hola" y "hola"
            dict_tags[tag] += 1
            dict_tag_word_count[tag][word] += 1

    # Cantidad de ocurrencias de palabras
    word_occurrences = sum(dict_words.values())

    # Cantidad de ocurrencias de tags
    tag_occurrences = sum(dict_tags.values())

    # Lista (tags, cantidad) ordenado de mayor a menor
    list_tags_descending = sorted(dict_tags.items(), key=lambda x: x[1], reverse=True)

    tag_max_frec_perc = [] # Lista de tuplas (tag, frecuencia, porcentaje)
    for tag, frec in list_tags_descending[:10]:
        percentage = float(frec)/tag_occurrences
        tag_max_frec_perc.append((tag, frec, percentage))

    print("Cantidad de oraciones =", count_sents)
    print("Cantidad de palabras =", len(dict_words), "=> Porcentajes")
    print("Cantidad de ocurrencias de palabras =", word_occurrences)
    print("Cantidad de tags =", len(dict_tags))

    list_word_count = []
    for tag, frec, perce in tag_max_frec_perc:
        list_word_count = sorted(dict_tag_word_count[tag].items(), key=lambda x: x[1], reverse=True)
        print(tag, frec, round(perce, 3), list_word_count[:5])
