from sentiment_analysis.task_01.preprocessing import delete_tildes
from sentiment_analysis.task_01.preprocessing import tweet_cleaner
from sentiment_analysis.task_01.preprocessing import remove_repeated
from sentiment_analysis.task_01.preprocessing import change_to_risas
from sentiment_analysis.task_01.preprocessing import remove_stopwords
from sentiment_analysis.task_01.preprocessing import tweet_stemming
from sentiment_analysis.task_01.preprocessing import spanish_stopwords

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

        # Emoticones Positivos
        self.positive_emoticons = {":-)", ":)", ":D", ":o)", ":]", "D:3",
                                   ":c)", ":>", "=]", "8)", "=)", ":}", ":^)",
                                   ":-D", "8-D", "8D", "x-D", "xD", "X-D",
                                   "XD", "=-D", "=D", "=-3", "=3", "B^D",
                                   ":')", ":*", ":-*", ":^*", ";-)", ";)",
                                   "*-)", "*)", ";-]", ";]", ";D", ";^)",
                                   ">:P", ":-P", ":P", "X-P", "x-p", "xp",
                                   "XP", ":-p", ":p", "=p", ":-b", ":b"}

        # Emoticones Negativos
        self.negative_emoticons = {">:[", ":-(", ":(", ":-c", ":-<", ":<",
                                   ":-[", ":[", ":{", ";(", ":-||", ">:(",
                                   ":'-(", ":'(", "D:<", "D=", "v.v"}

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
        # vectorizer = CountVectorizer(analyzer='word',
        #                              tokenizer=self.my_tokenizer,
        #                              lowercase=True,
        #                              stop_words=spanish_stopwords)
        vectorizer = CountVectorizer(analyzer='word',
                                     tokenizer=self.my_tokenizer)
        if name_vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word',
                                         tokenizer=self.my_tokenizer,
                                         lowercase=True,
                                         stop_words=spanish_stopwords)

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
        Tokenizador creado usando los metodos del modulo preprocessing.py
        """
        # IDEA: Buscar todos los emojis positivos y negativos y reemplazarlos
        #       con los string "positiveemoticon" y "negativeemoticon"
        #       Lo mismo con las palabras positivas y negativas:
        #       "positiveword" y "negativeword"
        tw = delete_tildes(tweet_content)
        tw = tweet_cleaner(tw)
        tw = remove_repeated(tw)
        tw = change_to_risas(tw)
        tw = self.tweet_tokenizer.tokenize(tw)
        tw = remove_stopwords(tw)
        tw = tweet_stemming(tw)

        return tw

    def emoticons_classify(self, tweet_content):
        """
        Pre-clasificacion de los tweets en base a los emoticones:
            * Si pos_emo = 0 y neg_emo = 0, el tweet es marcado como "NONE".
            * Si pos_emo = 0 y neg_emo > 0, el tweet es marcado como "N".
            * Si pos_emo > 0 y neg_emo > 0, el tweet es marcado como "NEU".
            * Si pos_emo > 0 y neg_emo = 0, el tweet es marcado como "P".
        """
        polarity_tag = {'NONE': 0, 'N': 1, 'NEU': 2, 'P': 3}

        classified_tweets = []
        for tw_c in tweet_content:
            tw = self.tweet_tokenizer.tokenize(tw_c)

            # Numero de emoticones positivos en el tweet
            pos_emo = len(self.positive_emoticons & set(tw))

            # Numero de emoticones negativos en el tweet
            neg_emo = len(self.negative_emoticons & set(tw))

            if pos_emo == 0 and neg_emo == 0:
                classified_tweets.append(polarity_tag.get('NONE', None))
            elif pos_emo == 0 and neg_emo > 0:
                classified_tweets.append(polarity_tag.get('N', None))
            elif pos_emo > 0 and neg_emo > 0:
                classified_tweets.append(polarity_tag.get('NEU', None))
            elif pos_emo > 0 and neg_emo == 0:
                classified_tweets.append(polarity_tag.get('P', None))

        return classified_tweets

    def classify_tweets(self, tweets_content):
        """
        Clasificamos los tweets usando la estimacion dada por el clasificador.
        """
        return self.pipeline.predict(tweets_content)
