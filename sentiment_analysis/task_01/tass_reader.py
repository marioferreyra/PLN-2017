# https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as ET


class CorpusTASSReader():

    def __init__(self, path, filename, is_corpus_tagged=True):
        """
        path -- Ruta en donde esta alojado el archivo XML
        filename -- Nombre del archivo XML
        is_corpus_tagged -- Flag que representa si el corpus esta taggeado con
                            sus respectivas polaridades
        """
        xml_file = path + "/" + filename
        tree = ET.parse(xml_file)  # Parseamos el XML en un arbol
        root = tree.getroot()  # Obtenemos la raiz del arbol (<tweets>)

        self.tweets_id = []  # Lista con los ID de los tweets
        self.tweets_content = []  # Lista con los Contenido de los tweets
        self.tweets_polarity = []  # Lista con las Polaridades de los tweets

        for tweet in root.iter('tweet'):
            tweet_id = int(tweet.find('tweetid').text)  # ID
            # user = tweet.find('user').text  # Usuario
            content = tweet.find('content').text  # Contenido
            # date = tweet.find('date').text  # Fecha
            # lang = tweet.find('lang').text  # Lenguaje

            # Si es el Corpus Train, guardo la polaridad
            if is_corpus_tagged:
                sentiment = tweet.find('sentiment')
                polarity = sentiment[0].find('value').text
                polarity = self.polarity_tagging(polarity)  # Polaridad
            # Si no, la polaridad es "NONE"
            else:
                polarity = "NONE"  # tweets sin polaridad => Polaridad = "NONE"
                polarity = self.polarity_tagging(polarity)

            # El tweet debe tener contenido para poder analizar su polaridad
            if content is not None:
                self.tweets_id.append(tweet_id)
                self.tweets_content.append(content)
                self.tweets_polarity.append(polarity)

        assert len(self.tweets_content) == len(self.tweets_polarity)

    def polarity_tagging(self, polarity):
        """
        Taggea un tweet segun su polaridad:
            * NONE = 0
            * N = 1
            * NEU = 2
            * P = 3
        """
        polarity_tag = {'NONE': 0, 'N': 1, 'NEU': 2, 'P': 3}

        return polarity_tag.get(polarity, None)

    def get_tweets_id(self):
        """
        Retorna una lista con los ID de los tweets.
        """
        return self.tweets_id

    def get_tweets_content(self):
        """
        Retorna una lista con los Contenidos de los tweets.
        """
        return self.tweets_content

    def get_tweets_polarity(self):
        """
        Retorna una lista con las Polaridades de los tweets.
        """
        return self.tweets_polarity
