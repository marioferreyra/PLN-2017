import xml.etree.ElementTree as ET
# https://docs.python.org/2/library/xml.etree.elementtree.html
# https://docs.python.org/3/library/xml.etree.elementtree.html


class Tweet:
    def __init__(self, tweet_id, user, date, lang, content, polarity):
        self.id = tweet_id
        self.user = user
        self.content = content
        self.date = date
        self.lang = lang
        self.polarity = polarity


def readXMLTrain(xml_file):
    tree = ET.parse(xml_file)  # Parseamos el XML en un arbol
    root = tree.getroot()  # Obtenemos la raiz del arbol (<tweets>)

    tweets = []  # Lista de tweets
    for tweet in root.iter('tweet'):
        tweet_id = int(tweet.find('tweetid').text)  # ID del tweet
        user = tweet.find('user').text  # Usuario del tweet
        content = tweet.find('content').text  # Contenido del tweet
        date = tweet.find('date').text  # Fecha del tweet
        lang = tweet.find('lang').text  # Lenguaje del tweet

        sentiment = tweet.find('sentiment')
        polarity = sentiment[0].find('value').text
        polarity = polarityTagging(polarity)  # Polaridad del tweet

        # El tweet debe tener contenido para poder analizar su polaridad
        if content is not None:
            tweet = Tweet(tweet_id, user, date, lang, content, polarity)
            tweets.append(tweet)

    return tweets


def readXMLTest(xml_file):
    tree = ET.parse(xml_file)  # Parseamos el XML en un arbol
    root = tree.getroot()  # Obtenemos la raiz del arbol (<tweets>)

    tweets = []  # Lista de tweets
    for tweet in root.iter('tweet'):
        tweet_id = int(tweet.find('tweetid').text)  # ID del tweet
        user = tweet.find('user').text  # Usuario del tweet
        content = tweet.find('content').text  # Contenido del tweet
        date = tweet.find('date').text  # Fecha del tweet
        lang = tweet.find('lang').text  # Lenguaje del tweet

        polatity = "NONE"  # A los tweets sin polaridad los taggeamos como NONE

        # El tweet debe tener contenido para poder analizar su polaridad
        if content is not None:
            tweet = Tweet(tweet_id, user, date, lang, content, polatity)
            tweets.append(tweet)

    return tweets


def polarityTagging(polarity):
    """
    Taggea un tweet segun su polaridad:
        * NONE = 0
        * N = 0
        * NEU = 0
        * P = 3
    """
    polarity_tag = {'NONE': 0, 'N': 1, 'NEU': 2, 'P': 3}

    return polarity_tag.get(polarity, None)
