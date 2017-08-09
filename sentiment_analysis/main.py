from read_xml import readXMLTrain, readXMLTest

from pprint import pprint

from collections import Counter

from preprocessing import tweet_cleaner, remove_stopwords, delete_tildes
from preprocessing import tweet_stemming, RepeatReplacer

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
spanish_stopwords = stopwords.words('spanish')  # Stopwords del español


# NOTA: Antes de usar remove_stopwords, tokenizar el contenido

my_replacer = RepeatReplacer()
positive_emoticons = [":-)", ":)", ":D", ":o)", ":]", "D:3", ":c)", ":>", "=]",
                      "8)", "=)", ":}", ":^)", ":-D", "8-D", "8D", "x-D", "xD",
                      "X-D", "XD", "=-D", "=D", "=-3", "=3", "B^D", ":')",
                      ":*", ":-*", ":^*", ";-)", ";)", "*-)", "*)", ";-]",
                      ";]", ";D", ";^)", ">:P", ":-P", ":P", "X-P", "x-p",
                      "xp", "XP", ":-p", ":p", "=p", ":-b", ":b"]

negative_emoticons = [">:[", ":-(", ":(", ":-c", ":-<", ":<", ":-[", ":[",
                      ":{", ";(", ":-||", ">:(", ":'-(", ":'(", "D:<", "D=",
                      "v.v"]

path = "/home/mario/Escritorio/PLN-2017/sentiment_analysis/Corpus_2017/Task_1"
train_file = "tw_faces4tassTrain1000rc.xml"
test_file = "TASS2017_T1_test.xml"

path02 = "/home/mario/Escritorio/PLN-2017/sentiment_analysis"
file_01 = "entrenador2.xml"
file_02 = "prueba2.xml"

train_tweets = readXMLTrain(path02, file_01)
test_tweets = readXMLTest(path02, file_02)
# Solo hay 3 tweets en este XML (rango 0, 1, 2)

# print(train_tweets[0].content)
# print(test_tweets[0].content)

def mi_tokenizador(content):
    """
    Mi tokenizador
    """
    tw = delete_tildes(content)
    tw = tweet_cleaner(tw)
    tw = TweetTokenizer().tokenize(tw)
    tw = remove_stopwords(tw)
    tw = tweet_stemming(tw)

    return tw

# print(mi_tokenizador(train_tweets[0].content), train_tweets[0].polarity)
# print(mi_tokenizador(train_tweets[1].content), train_tweets[1].polarity)
# print(mi_tokenizador(train_tweets[2].content), train_tweets[2].polarity)
# print("=====================================")
# print(mi_tokenizador(test_tweets[0].content))
# print(mi_tokenizador(test_tweets[1].content))
# print(mi_tokenizador(test_tweets[2].content))

polarity_labels = [tweet.polarity for tweet in train_tweets]
train_tweets_cleaned = [tweet.content for tweet in train_tweets]

test_tweets_cleaned = [tweet.content for tweet in test_tweets]
# print(test_tweets_cleaned)
# tweets_cleaned = []
# for tweet in train_tweets:
#     t = mi_tokenizador(tweet.content)
#     tweets_cleaned.append(t)


# pprint(polarity_labels)
# pprint(tweets_cleaned)

# Vectorizador
v = CountVectorizer(analyzer='word',
                    tokenizer=mi_tokenizador,
                    lowercase=True,
                    stop_words=spanish_stopwords)

# Clasificador
c = LinearSVC()


pipe = Pipeline([("vectorizador", v), ("clasificador", c)])

pipe.fit(train_tweets_cleaned, polarity_labels)


resultados = pipe.predict(test_tweets_cleaned)

print(resultados)
counter = Counter(resultados)  # Cantidad de polaridades
print(counter, sum(counter.values()))
print("NONE = {}".format(counter.get(0, None)))
print("N = {}".format(counter.get(1, None)))
print("NEU = {}".format(counter.get(2, None)))
print("P = {}".format(counter.get(3, None)))

# print("Tweet")
# print("=======")

# print("Contenido")
# print("=========")
# my_content = tweet.content
# print(my_content)
# if tweet.polarity == 0:
#     print("NONE")
# elif tweet.polarity == 1:
#     print("NEGATIVO")
# elif tweet.polarity == 2:
#     print("NEUTRO")
# elif tweet.polarity == 3:
#     print("POSITIVO")

# print("\nContenido, con las tildes eliminadas")
# print("====================================")
# my_content = delete_tildes(my_content)
# print(my_content)

# print("\nContenido, limpiado")
# print("===================")
# my_content = tweet_cleaner(my_content)
# print(my_content)

# print("\nContenido Tokenizando")
# print("=====================")
# my_content = TweetTokenizer().tokenize(my_content)
# print(my_content)

# print("\nContenido, removiendo Stopwords")
# print("===============================")
# my_content = remove_stopwords(my_content)
# print(my_content)
