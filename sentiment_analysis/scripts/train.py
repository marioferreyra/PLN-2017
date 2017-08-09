"""
Train a list of tweets.

Usage:
  train.py -o <file>
  train.py -h | --help

Options:
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from sentiment_analysis.read_xml import readXMLTrain
from sentiment_analysis.sentiment import TwitterPolarity


if __name__ == '__main__':
    opts = docopt(__doc__)

    path = "/home/mario/Escritorio/PLN-2017/sentiment_analysis/Corpus_2017/Task_1"
    file = "tw_faces4tassTrain1000rc.xml"
    tweets = readXMLTrain(path, file)

    # Train the model
    print("Training model ...")
    model = TwitterPolarity(tweets)
    print("Finished")

    # Save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
