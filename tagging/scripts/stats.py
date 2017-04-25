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


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the data
    # corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    PATH = "/home/mario/Escritorio/ancora-2.0"
    corpus = SimpleAncoraCorpusReader(PATH)
    sents = list(corpus.tagged_sents())

    # print(sents[0])
    # Compute the statistics
    # print('sents: {}'.format(len(sents)))
    count_sents = len(sents) # Cantidad de oraciones
    dict_words = defaultdict(int) # Cantidad de palabras
    dict_tags = defaultdict(int) # Cantidad de tags

    for sent in sents:
        # sent es una lista de pares: (palabra, tag)
        for word, tag in sent:
            dict_words[word.lower()] += 1 # Para no diferenciar entre "Hola" y "hola"
            dict_tags[tag] += 1

    count_occurrences = sum(dict_words.values()) # Cantidad de ocurrencias de palabras
    count_tags = sum(dict_tags.values()) # Cantidad de ocurrencias de tags
    # Lista de (tags, cantidad) ordenado de mayor a menor
    list_tags_descending = sorted(dict_tags.items(), key=lambda x: x[1], reverse=True)

    # tag_max_frec = list_tags_descending[:10]
    tag_max_frec_perc = [] # Lista de tuplas (tag, frecuencia, porcentaje)
    # for tag, frec in tag_max_frec:
    for tag, frec in list_tags_descending[:10]:
        percentage = float(frec)/count_tags
        tag_max_frec_perc.append((tag, frec, percentage))

    print("Cantidad de oraciones =", count_sents)
    print("Cantidad de palabras =", len(dict_words))
    print("Cantidad de ocurrencias =", count_occurrences)
    print("Cantidad de tags =", len(dict_tags))

    for t, f, p in tag_max_frec_perc:
        print(t, f, p)
