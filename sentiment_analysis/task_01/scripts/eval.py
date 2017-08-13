"""
Evaulate a model.

Usage:
  eval.py -i <file> -r <rst>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -r <rst>      Name for file with results
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from sentiment_analysis.task_01.read_xml import readXMLTest
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
import random


def create_file_result(filename, list_id, list_polarity):
    """
    Crea el archivo con los resultados que se obtuvieron con el modelo
    entrenado.
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

    print("Evaluating model")
    model_classify_tweets = model.classify_tweets(tweets_content)

    counter = Counter(model_classify_tweets)  # Cantidad de polaridades

    print(counter)
    print("Tweets analizados: {}".format(sum(counter.values())))
    print("\t* NONE = {}".format(counter.get(0, None)))
    print("\t* N = {}".format(counter.get(1, None)))
    print("\t* NEU = {}".format(counter.get(2, None)))
    print("\t* P = {}".format(counter.get(3, None)))

    y_test = [random.randint(0, 3) for _ in range(1899)]  # RANDOM NO CORRECTO

    accuracy = accuracy_score(y_test, model_classify_tweets)
    # precision = precision_score(y_test, model_classify_tweets)
    # recall = recall_score(y_test, model_classify_tweets)
    # f1 = f1_score(y_test, model_classify_tweets)

    print("")
    print("Accuracy = {}%".format(round(accuracy * 100, 2)))
    # print("Precision = {}%".format(round(precision * 100, 2)))
    # print("Recall = {}%".format(round(recall * 100, 2)))
    # print("F1 = {}%".format(round(f1 * 100, 2)))

    # Creamos archivo con los resultados
    tweets_id = [tweet.id for tweet in tweets_list]
    r = opts['-r']
    create_file_result(r, tweets_id, model_classify_tweets)
