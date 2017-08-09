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
from sentiment_analysis.read_xml import readXMLTest
from collections import Counter


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # Load the data
    path = "/home/mario/Escritorio/PLN-2017/sentiment_analysis/Corpus_2017/Task_1/"
    file = "TASS2017_T1_test.xml"
    tweets = readXMLTest(path, file)

    tweets_content = [tweet.content for tweet in tweets]

    print("Evaluating model")
    model_classify_tweets = model.classify_tweets(tweets_content)

    counter = Counter(model_classify_tweets)  # Cantidad de polaridades

    print(counter, sum(counter.values()))
    print("NONE = {}".format(counter.get(0, None)))
    print("N = {}".format(counter.get(1, None)))
    print("NEU = {}".format(counter.get(2, None)))
    print("P = {}".format(counter.get(3, None)))
