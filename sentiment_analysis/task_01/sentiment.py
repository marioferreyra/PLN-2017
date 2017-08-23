from sentiment_analysis.task_01.preprocessing import PreprocessingTweet

from nltk.tokenize import TweetTokenizer

# Vectorizadores
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Clasificadores
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Pipeline
from sklearn.pipeline import Pipeline


class TwitterPolarity:

    def __init__(self,
                 tweets_content,
                 tweets_polarity,
                 name_vectorizer="count",
                 name_classifier="svc"):
        """
        tweets_content -- Lista con los Contenidos de cada tweet.
        tweets_polarity -- Lista con las Polaridades de cada tweet.
        name_vectorizer -- Tipo de vectorizador a usar
                           [default: CountVectorizer]
        name_classifier -- Tipo de Clasificador a usar
                           [default: LinearSVC]
        """
        self.name_vectorizer = name_vectorizer  # Nombre del Vectorizador
        self.name_classifier = name_classifier  # Nombre del Clasificador

        self.preprocessing = PreprocessingTweet()

        # Emoticones Positivos
        self.positive_emoticons = self.preprocessing.get_positive_emoticons()

        # Emoticones Negativos
        self.negative_emoticons = self.preprocessing.get_positive_emoticons()

        # Stopwords del español
        self.spanish_stopwords = self.preprocessing.get_spanish_stopwords()

        # Tokenizador de tweets
        self.tweet_tokenizer = TweetTokenizer()

        # Vectorizador
        v = self.select_vectorizer(name_vectorizer)

        # Clasificador
        c = self.select_classifier(name_classifier)

        pipe = Pipeline([("vectorizador", v), ("clasificador", c)])

        pipe.fit(tweets_content, tweets_polarity)

        self.pipeline = pipe

    def select_vectorizer(self, name_vectorizer):
        """
        Elije el vectorizador a usar.
        """
        vectorizer = CountVectorizer(analyzer='word',
                                     tokenizer=self.my_tokenizer,
                                     lowercase=True,
                                     stop_words=self.spanish_stopwords)
        if name_vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word',
                                         tokenizer=self.my_tokenizer,
                                         lowercase=True,
                                         stop_words=self.spanish_stopwords)

        return vectorizer

    def select_classifier(self, name_classifier):
        """
        Elije el clasificador a usar.
        """
        classifier = LinearSVC()
        if name_classifier == "logreg":
            classifier = LogisticRegression()
        elif name_classifier == "forest":
            classifier = RandomForestClassifier()

        return classifier

    def get_names_vectorizer_classifier(self):
        """
        Me devuelve el nombre del Vectorizador y del Clasificador usados.
        """
        vectorizers = {"count": "CountVectorizer",
                       "tfidf": "TfidfVectorizer"}

        classifiers = {"svc": "LinearSVC",
                       "logreg": "LogisticRegression",
                       "forest": "RandomForestClassifier"}

        name_vec = vectorizers.get(self.name_vectorizer, None)
        name_clas = classifiers.get(self.name_classifier, None)

        return name_vec, name_clas

    def my_tokenizer(self, tweet_content):
        """
        Tokenizador creado usando los metodos de la clase PreprocessingTweet
        """
        tw = self.preprocessing.delete_tildes(tweet_content)
        tw = self.preprocessing.change_pos_neg_words_emojis(tw)
        tw = self.preprocessing.tweet_cleaner(tw)
        tw = self.preprocessing.remove_repeated(tw)
        tw = self.preprocessing.change_to_risas(tw)
        tw = self.tweet_tokenizer.tokenize(tw)
        tw = self.preprocessing.remove_stopwords(tw)
        tw = self.preprocessing.tweet_stemming(tw)

        return tw

    def emoticons_classify(self, tweet_content):
        """
        Pre-clasificacion de los tweets en base a los emoticones:
            * Si pos_emo = 0 y neg_emo = 0, el tweet es marcado como "NONE".
            * Si pos_emo = 0 y neg_emo > 0, el tweet es marcado como "N".
            * Si pos_emo > 0 y neg_emo > 0, el tweet es marcado como "NEU".
            * Si pos_emo > 0 y neg_emo = 0, el tweet es marcado como "P".
        """
        polarity_emo = []
        for tw_c in tweet_content:
            tw = self.tweet_tokenizer.tokenize(tw_c)

            # Numero de emoticones positivos en el tweet
            pos_emo = len(self.positive_emoticons & set(tw))

            # Numero de emoticones negativos en el tweet
            neg_emo = len(self.negative_emoticons & set(tw))

            if pos_emo == 0 and neg_emo == 0:
                polarity_emo.append("NONE")
            elif pos_emo == 0 and neg_emo > 0:
                polarity_emo.append("N")
            elif pos_emo > 0 and neg_emo > 0:
                polarity_emo.append("NEU")
            elif pos_emo > 0 and neg_emo == 0:
                polarity_emo.append("P")

        return polarity_emo

    def classify_tweets(self, tweets_content):
        """
        Clasificamos los tweets usando la estimacion dada por el clasificador.
        """
        return self.pipeline.predict(tweets_content)
