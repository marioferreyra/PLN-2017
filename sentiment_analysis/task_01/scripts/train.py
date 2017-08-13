"""
Train a list of tweets.

Usage:
  train.py [-v <vec>] [-c <clf>] -o <file>
  train.py -h | --help

Options:
  -v <vec>      Vectorizer to use [default: count]
                  count: CountVectorizer
                  tfidf: TfidfVectorizer
  -c <clf>      Classifier to use [default: svc]
                  svc: LinearSVC
                  logreg: LogisticRegression
                  forest: RandomForestClassifier
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
import pickle
from docopt import docopt
from sentiment_analysis.task_01.read_xml import readXMLTrain
from sentiment_analysis.task_01.sentiment import TwitterPolarity


if __name__ == '__main__':
    opts = docopt(__doc__)

    path = "/home/mario/Escritorio/PLN-2017/sentiment_analysis/task_01/Corpus"
    file = "tw_faces4tassTrain1000rc.xml"
    tweets_list = readXMLTrain(path, file)

    # Select Vectorizer
    v = opts['-v']
    if v == "tfidf":
        print("### Vectorizer: TfidfVectorizer ###")
    else:
        print("### Vectorizer: CountVectorizer ###")

    # Select Classifier
    c = opts['-c']
    if c == "logreg":
        print("### Classifier: LogisticRegression ###")
    elif c == "forest":
        print("### Classifier: RandomForestClassifier ###")
    else:
        print("### Classifier: LinearSVC ###")

    # Train the model
    print("Training model ...")
    model = TwitterPolarity(tweets_list, v, c)
    print("Finished")

    # Save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
