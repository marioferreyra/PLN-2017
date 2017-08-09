# from pprint import pprint

from sentiment_analysis.preprocessing import tweet_cleaner, remove_stopwords, delete_tildes
from sentiment_analysis.preprocessing import tweet_stemming, RepeatReplacer

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords


class TwitterPolarity:

    def __init__(self, tweets_list):
        """
        Init
        """
        polarity_list = [tweet.polarity for tweet in tweets_list]
        content_tweets = [tweet.content for tweet in tweets_list]

        spanish_stopwords = stopwords.words('spanish')  # Stopwords español

        # Vectorizador
        v = CountVectorizer(analyzer='word',
                            tokenizer=self.tweet_tokenizer,
                            lowercase=True,
                            stop_words=spanish_stopwords)

        c = LinearSVC()  # Clasificador

        pipe = Pipeline([("vectorizador", v), ("clasificador", c)])

        pipe.fit(content_tweets, polarity_list)

        self.pipeline = pipe

    def tweet_tokenizer(self, content):
        """
        Mi tokenizador
        """
        tw = delete_tildes(content)
        tw = tweet_cleaner(tw)
        tw = TweetTokenizer().tokenize(tw)
        tw = remove_stopwords(tw)
        tw = tweet_stemming(tw)

        return tw

    def classify_tweets(self, content):
        """
        Predecimos los tweets
        """
        return self.pipeline.predict(content)
