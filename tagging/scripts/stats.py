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
from collections import defaultdict


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargamos el Ancora
    PATH = "/home/mario/Escritorio/ancora-3.0.1es"
    corpus = SimpleAncoraCorpusReader(PATH)
    sents = list(corpus.tagged_sents())

    # Compute the statistics
    dict_words = defaultdict(int)
    dict_tags = defaultdict(int)

    # tag : {word : cantidad}
    dict_tag_word_count = defaultdict(lambda: defaultdict(int))

    # word : {tag : cantidad}
    # dict_word_tag = defaultdict(set)
    dict_word_tag_count = defaultdict(lambda: defaultdict(int))

    for sent in sents:
        # sent es una lista de pares: (palabra, tag)
        for word, tag in sent:
            dict_words[word] += 1
            dict_tags[tag] += 1
            dict_tag_word_count[tag][word] += 1
            # dict_word_tag[word].add(tag)
            dict_word_tag_count[word][tag] += 1

    count_sents = len(sents)  # Cantidad de oraciones
    count_words = len(dict_words)  # Cantidad de palabras (word types)
    count_tags = len(dict_tags)  # Cantidad de tags

    word_occurrences = sum(dict_words.values())  # Cantidad de tokens
    tag_occurrences = sum(dict_tags.values())  # Cantidad de ocurrencias de tag

    # Lista (tags, cantidad) ordenado de mayor a menor
    list_tags_descending = sorted(dict_tags.items(), key=lambda x: x[1],
                                  reverse=True)

    # Tomamos los 10 tags mas frecuentes
    tags_most_frec = list_tags_descending[:10]

    # Lista de tuplas (tag, frecuencia, porcentaje)
    list_tag_frec_perc = []
    for tag, frec in tags_most_frec:
        percentage = (float(frec)/tag_occurrences)*100
        list_tag_frec_perc.append((tag, frec, round(percentage, 2)))

    print("Estadisticas")
    print("============")
    print("Cantidad de oraciones = {}".format(count_sents))
    print("Cantidad de tokens = {}".format(count_words))
    print("Cantidad de palabras = {}".format(word_occurrences))
    print("Cantidad de tags = {}".format(count_tags))
    print("\n")

    print("--------------------------------------------------------------")
    print(" {:^7} | {} | {} | {} ".format("Tag",
                                          "Frecuencia",
                                          "Porcentaje",
                                          "5 Palabras mas frecuentes"))
    print("--------------------------------------------------------------")
    list_word_count = []
    for tag, frec, perce in list_tag_frec_perc:
        list_word_count = sorted(dict_tag_word_count[tag].items(),
                                 key=lambda x: x[1], reverse=True)
        words_most_frec = list_word_count[:5]
        new_list = " ".join([word for word, _ in words_most_frec])
        print("{:^8} | {:^10} | {:^9}% | {}".format(tag,
                                                    frec,
                                                    perce,
                                                    new_list))

    print("\n")
    print("---------------------------------------\
----------------------------------")
    print(" {} | {} | {} | {} ".format("Nivel de Ambigüedad",
                                       "#Palabras",
                                       "Porcentaje",
                                       "5 Palabras mas frecuentes"))
    print("---------------------------------------\
----------------------------------")
    # Niveles de ambiguedad: 1 ... 9
    for level in range(1, 10):
        # Lista de tuplas (palabra, ocurrencias)
        list_word_count = []
        for word, tag_count in dict_word_tag_count.items():
            # Palabras con nivel de ambiguedad 'level'
            if len(tag_count.items()) == level:
                list_word_count.append((word, dict_words[word]))

        # Ordenamos mayor a menor segun la cantidad de ocurrencias
        list_word_count = sorted(list_word_count, key=lambda x: x[1],
                                 reverse=True)

        # Obtenemos solo las palabras
        list_words = [word for word, _ in list_word_count]
        length_list_words = len(list_words)

        # Porcentaje de ocurrencias de palabras sobre el total
        percentage = round((float(length_list_words)/count_words)*100, 2)
        new_list = " ".join(list_words[:5])
        print("{:^20} | {:^9} | {:^9}% | {}".format(level,
                                                    length_list_words,
                                                    percentage, new_list))
