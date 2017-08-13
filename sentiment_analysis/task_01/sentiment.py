from sentiment_analysis.task_01.preprocessing import delete_tildes
from sentiment_analysis.task_01.preprocessing import tweet_cleaner
from sentiment_analysis.task_01.preprocessing import remove_repeated
from sentiment_analysis.task_01.preprocessing import change_to_risas
from sentiment_analysis.task_01.preprocessing import remove_stopwords
from sentiment_analysis.task_01.preprocessing import tweet_stemming

from nltk.tokenize import TweetTokenizer

# Vectorizadores
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Clasificadores
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline


from nltk.corpus import stopwords

# Recordatorio: Ver se usar cross validation para determinar los mejores
# parametros del clasificador


class TwitterPolarity:

    def __init__(self,
                 tweets_list,
                 name_vectorizer="count",
                 name_classifier="svc"):
        """
        Init
        """
        self.tweets_id = [tweet.id for tweet in tweets_list]
        self.tweets_content = [tweet.content for tweet in tweets_list]
        self.tweets_polarity = [tweet.polarity for tweet in tweets_list]

        # Tokenizador de tweets
        self.tweet_tokenizer = TweetTokenizer()

        # Vectorizador
        v = self.select_vectorizer(name_vectorizer)

        # Clasificador
        c = self.select_classifier(name_classifier)

        pipe = Pipeline([("vectorizador", v), ("clasificador", c)])

        pipe.fit(self.tweets_content, self.tweets_polarity)

        self.pipeline = pipe

    def select_vectorizer(self, name_vectorizer):
        """
        Elije el vectorizador a usar
        """
        spanish_stopwords = stopwords.words('spanish')  # Stopwords del español

        vectorizer = CountVectorizer(analyzer='word',
                                     tokenizer=self.my_tokenizer,
                                     lowercase=True,
                                     stop_words=spanish_stopwords)
        if name_vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word',
                                         tokenizer=self.my_tokenizer,
                                         lowercase=True,
                                         stop_words=spanish_stopwords)

        return vectorizer

    def select_classifier(self, name_classifier):
        """
        Elije el clasificador a usar
        """
        classifier = LinearSVC()
        if name_classifier == "logreg":
            classifier = LogisticRegression()
        elif name_classifier == "forest":
            classifier = RandomForestClassifier()

        return classifier

    def my_tokenizer(self, content):
        """
        Mi tokenizador
        """
        tw = delete_tildes(content)
        tw = tweet_cleaner(tw)
        tw = remove_repeated(tw)
        tw = change_to_risas(tw)
        tw = self.tweet_tokenizer.tokenize(tw)
        tw = remove_stopwords(tw)
        tw = tweet_stemming(tw)

        return tw

    def classify_tweets(self, content):
        """
        Predecimos los tweets
        """
        return self.pipeline.predict(content)

    def emoticons_classify(self, content):
        """
        Clasificacion en base a los emoticones.
        """
        pass
