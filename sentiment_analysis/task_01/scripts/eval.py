"""
Evaluate a model with a list of tweets.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from sentiment_analysis.task_01.tass_reader import CorpusTASSReader
from collections import Counter
from sklearn.metrics import confusion_matrix


def get_accuracy(polarity_model, polarity_gold):
    """
    Calcula la Accuracy:
        accuracy = total_aciertos / total_casos
    Otra forma es usar:
        from sklearn.metrics import accuracy_score
        accuracy_score(polarity_model, polarity_gold)
    """
    # hits = 0
    # for pm, pg in zip(polarity_model, polarity_gold):
    #     if pm == pg:
    #         hits += 1
    # acc = hits / len(polarity_gold)
    assert len(polarity_model) == len(polarity_gold)

    hits_polarity = [pm == pg for pm, pg in zip(polarity_model, polarity_gold)]
    hits = sum(hits_polarity)
    total = len(polarity_gold)
    acc = hits / total
    # VER DE DEVOLVER (hits / total)

    return acc


def get_precision(hits, polarity_model):
    """
    Calcula la Precision.
    """
    return hits / polarity_model


def get_recall(hits, polarity_gold):
    """
    Calcula la Recall.
    """
    return hits / polarity_model


def get_f1(precision, recall):
    """
    Calcula la Recall:
        f1 = 2 * (precision * recall) / (precision + recall)
    Otra forma es usar:
        from sklearn.metrics import recall_score
    """
    if precision + recall > 0.0:
        result = 2 * (precision * recall) / (precision + recall)
    else:
        result = 0.0

    return result


def heuristic_classification(classified_tweets, classified_tweets_emoticons):
    """
    La polaridad se define por la siguiente regla:
     * Usamos como preclasificacion a la dada por el uso de los emoticones.
     * Mantenemos la polaridad predefida si el tweet es marcado como "P" o "N"
       de lo contrario tomamos el valor estimado por el clasificador.
    """
    assert len(classified_tweets) == len(classified_tweets_emoticons)

    classified_heuristic_rule = []
    for tw_c, tw_e in zip(classified_tweets, classified_tweets_emoticons):
        # Clasificacion usando emoticones es "N" o "P"
        if tw_e in {1, 3}:
            classified_heuristic_rule.append(tw_e)
        # Clasificacion usando emoticones es "NONE" o "NEU"
        else:
            classified_heuristic_rule.append(tw_c)

    assert len(classified_heuristic_rule) == len(classified_tweets)

    return classified_heuristic_rule


def print_results(polarity_model, polarity_gold):
    """
    Imprime los resutados obtenidos.
    """
    counter_results = Counter(polarity_model)

    print("| Polaridad | Cantidad de Tweets |")
    print("|:---------:|:------------------:|")
    print("| {:^9} | {:^18} |".format("NONE", counter_results.get(0, 0)))
    print("| {:^9} | {:^18} |".format("N", counter_results.get(1, 0)))
    print("| {:^9} | {:^18} |".format("NEU", counter_results.get(2, 0)))
    print("| {:^9} | {:^18} |".format("P", counter_results.get(3, 0)))

    accuracy = get_accuracy(polarity_model, polarity_gold)

    print("")
    print("* Accuracy = {}%".format(round(accuracy * 100, 2)))
    print("")


def create_file_result(filename, list_id, list_polarity):
    """
    Crea un archivo txt con los resultados de las polaridades obtenidas.
    El formato es el siguiente:
        tweet_id \t polarity
    """
    assert len(list_id) == len(list_polarity)

    decode_polarity = {0: 'NONE', 1: 'N', 2: 'NEU', 3: 'P'}
    directory_direction = "/home/mario/Escritorio/PLN-2017/sentiment_analysis\
/task_01/Results/"
    path = directory_direction + filename + ".txt"
    f = open(path, "w")

    for my_id, my_polarity in zip(list_id, list_polarity):
        my_polarity = decode_polarity.get(my_polarity, None)
        f.write(str(my_id) + "\t" + my_polarity + "\n")

    f.close()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the model
    filename = opts['-i']
    model_path = "Models/" + filename
    f = open(model_path, 'rb')
    model = pickle.load(f)
    f.close()

    # Recuperamos el nombre del Vectorizador y Clasificador que se usaron
    n_vec, n_clas = model.get_names_vectorizer_classifier()

    # Cargamos los Tweets del Corpus Development
    path = "/home/mario/Escritorio/PLN-2017/sentiment_analysis/task_01/Corpus"
    file = "TASS2017_T1_development.xml"
    corpus_reader = CorpusTASSReader(path, file, is_corpus_tagged=True)
    tweets_id = corpus_reader.get_tweets_id()
    tweets_content = corpus_reader.get_tweets_content()
    polarity_gold = corpus_reader.get_tweets_polarity()

    # Cantidad de tweets que se van a clasificar
    print("* Tweets analizados: {}".format(len(tweets_content)))

    # ============================
    # Preclasificacion usando Emoticones
    # polarity_emo = model.emoticons_classify(tweets_content)
    # accuracy_emoticons = accuracy_score(polarity_emo, polarity_gold)

    # print("\n### Preclasificación usando Emoticones")
    # print_results(Counter(polarity_emo), accuracy_emoticons)

    # ==============================
    # Evaluacion usando clasificador
    polarity_model = model.classify_tweets(tweets_content)
    print("\n### Vectorizador *\"{}\"* y Clasificador *\"{}\"*".format(n_vec,
                                                                       n_clas))
    print_results(polarity_model, polarity_gold)

    # ===========================================
    # Evaluacion usando emoticones y clasificador
    # classified_heuristic = heuristic_classification(polarity_model,
    #                                                 polarity_emo)
    # accuracy_heuristic = accuracy_score(classified_heuristic, polarity_gold)

    # print("\n##### Clasificación en base a Heuristica")
    # print_results(Counter(classified_heuristic), accuracy_heuristic)

    # # ===================================================
    # # Creamos archivo con los resultados del clasificador
    # r = opts['-r']
    # create_file_result(r, tweets_id, polarity_model)
