"""
Evaluate a model with a list of tweets.

Usage:
  eval.py -i <file> -r <rst>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -r <rst>      Name for file with results
  -h --help     Show this screen.
"""
import pickle
import random
from docopt import docopt
from sentiment_analysis.task_01.read_xml import readXMLTest
from collections import Counter
from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score, recall_score, f1_score


def estimate_accuracy(classified_tweets, simulations):
    """
    Estimando la Accuracy simulando con distintos Golden Polarity
    (esto es porque el Corpus Test no tiene anotadas las polaridades).

    NOTA: Esto no es lo mejor.
    """
    # Simulamos unas "simulations" veces
    a = 0
    for _ in range(simulations):
        # 1899 es la cantidad de tweets en Corpus Test
        golden_polarity = [random.randint(0, 3) for _ in range(1899)]
        a += accuracy_score(golden_polarity, classified_tweets)

    accuracy = float(a) / simulations

    return accuracy


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


def print_results(counter_results, accuracy):
    """
    Imprime los resutados obtenidos.
    """
    print("Tweets analizados: {}".format(sum(counter_results.values())))
    print("\t* NONE = {}".format(counter_results.get(0, None)))
    print("\t* N = {}".format(counter_results.get(1, None)))
    print("\t* NEU = {}".format(counter_results.get(2, None)))
    print("\t* P = {}".format(counter_results.get(3, None)))
    print("\t* Accuracy = {}%".format(round(accuracy * 100, 2)))


def create_file_result(filename, list_id, list_polarity):
    """
    Crea un archivo txt con los resultados de las polaridades obtenidas.
    El formato es el siguiente:
        tweet_id \t polarity
    """
    assert len(list_id) == len(list_polarity)

    decode_polarity = {0: 'NONE', 1: 'N', 2: 'NEU', 3: 'P'}
    directory_direction = "/home/mario/Escritorio/PLN-2017/sentiment_analysis\
/task_01/Resultados/"
    path = directory_direction + filename + ".txt"
    f = open(path, "w")

    for my_id, my_polarity in zip(list_id, list_polarity):
        my_id = str(my_id)
        my_polarity = decode_polarity.get(my_polarity, None)
        f.write(my_id + "\t" + my_polarity + "\n")

    f.close()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # Load the data
    path = "/home/mario/Escritorio/PLN-2017/sentiment_analysis/task_01/Corpus"
    file = "TASS2017_T1_test.xml"
    tweets_list = readXMLTest(path, file)
    tweets_content = [tweet.content for tweet in tweets_list]

    # ============================
    # Evaluacion usando emoticones
    classified_tweets_emoticons = model.emoticons_classify(tweets_content)
    counter_emoticons = Counter(classified_tweets_emoticons)
    accuracy_emoticons = estimate_accuracy(classified_tweets_emoticons, 1000)

    print("\nPreclasificacion usando Emoticones")
    print("==================================")
    print_results(counter_emoticons, accuracy_emoticons)

    # ==============================
    # Evaluacion usando clasificador
    classified_tweets = model.classify_tweets(tweets_content)
    counter = Counter(classified_tweets)
    accuracy = estimate_accuracy(classified_tweets, 1000)

    print("\nClasificacion usando Clasificador")
    print("=================================")
    print_results(counter, accuracy)

    # ===========================================
    # Evaluacion usando emoticones y clasificador
    classified_tw_emo = classified_tweets_emoticons
    classified_heuristic_rule = heuristic_classification(classified_tweets,
                                                         classified_tw_emo)
    counter_heuristic = Counter(classified_heuristic_rule)
    accuracy_heuristic = estimate_accuracy(classified_heuristic_rule, 1000)

    print("\nClasificacion usando Clasificador y Emoticones")
    print("==============================================")
    print_results(counter_heuristic, accuracy_heuristic)

    # ===================================================
    # Creamos archivo con los resultados del clasificador
    tweets_id = [tweet.id for tweet in tweets_list]
    r = opts['-r']
    create_file_result(r, tweets_id, classified_tweets)
