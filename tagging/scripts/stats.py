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
    PATH = "/home/mario/Escritorio/ancora-3.0.1es"
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
    dict_tag_word_count = defaultdict(lambda : defaultdict(int))

    # word : {tag : cantidad}
    # dict_word_tag = defaultdict(set)
    dict_word_tag_count = defaultdict(lambda : defaultdict(int))

    for sent in sents:
        # sent es una lista de pares: (palabra, tag)
        for word, tag in sent:
            dict_words[word] += 1
            dict_tags[tag] += 1
            dict_tag_word_count[tag][word] += 1
            # dict_word_tag[word].add(tag)
            dict_word_tag_count[word][tag] +=1

    # Cantidad de ocurrencias de palabras (cantidad de tokens)
    word_occurrences = sum(dict_words.values())

    # Cantidad de ocurrencias de tags
    tag_occurrences = sum(dict_tags.values())

    # Lista (tags, cantidad) ordenado de mayor a menor
    list_tags_descending = sorted(dict_tags.items(), key=lambda x: x[1], reverse=True)

    # Tomamos los 10 tags mas frecuentes
    list_tags_most_frec = list_tags_descending[:10]

    # Lista de tuplas (tag, frecuencia, porcentaje)
    list_tag_frec_perc = []
    for tag, frec in list_tags_most_frec:
        percentage = float(frec)/tag_occurrences
        list_tag_frec_perc.append((tag, frec, percentage))

    print("Estadisticas\n============")
    print("Cantidad de oraciones = {:>19}".format(count_sents)) # BIEN
    print("Cantidad de tags = {:>21}".format(len(dict_tags))) # BIEN
    print("Cantidad de ocurrencias de palabras = {}".format(word_occurrences)) # BIEN
    print("Cantidad de palabras distintas = {:>10}".format(len(dict_words))) # BIEN
    print("\n")

    print("-----------------------------------------------------------")
    print(" TAG | FRECUENCIA | PORCENTAJE | 5 PALABRAS MAS FRECUENTES ")
    print("-----------------------------------------------------------")
    list_word_count = []
    for tag, frec, perce in list_tag_frec_perc:
        list_word_count = sorted(dict_tag_word_count[tag].items(), key=lambda x: x[1], reverse=True)

        list_words_most_frec = list_word_count[:5]
        # new_list = [word_count[0] for word_count in list_words_most_frec]
        new_list = [word for word, _ in list_words_most_frec]
        print("{:^8} | {:^5} | {:^5} | {}".format(tag, frec, round(perce, 3), new_list))

    print("\n")
    print("--------------------------------------------------------------------------")
    # print("{:^10} | {:^10} | {:^10} | {:^10}".format("Nivel de Ambigüedad", "#Palabras", "Porcentaje", "5 Palabras mas frecuentes"))
    print(" Nivel de Ambigüedad | #Palabras | Porcentaje | 5 Palabras mas frecuentes ")
    print("--------------------------------------------------------------------------")
    for level in range(1, 10):
        new_list_word_count = []
        for word, tag_count in dict_word_tag_count.items():
            my_tags = tag_count.items()
            if len(my_tags) == level:
                new_list_word_count.append((word, dict_words[word]))
        new_list_word_count = sorted(new_list_word_count, key=lambda x: x[1], reverse=True)
        new_list_word = [word for word, _ in new_list_word_count]
        length_list = len(new_list_word)
        porcentaje = round((float(length_list)/word_occurrences)*100, 5)
        print("{:^20} | {:^9} | {:^9}% | {}".format(level, length_list, porcentaje, new_list_word[:5]))
