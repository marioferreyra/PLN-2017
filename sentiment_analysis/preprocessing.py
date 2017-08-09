import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descargar las stopwords usando el comando nltk.download()
spanish_stopwords = stopwords.words('spanish')  # Stopwords del español

stemmer = SnowballStemmer("spanish")

def tweet_cleaner(content):
    """
    Limpiamos un tweet.
    """
    # Convertimos el texto a minusculas
    new_content = content.lower()

    # Remueve URL signos de puntuación, nombres de usuario o cualquier
    # caracteres no alfanuméricos.También separa la palabra con un solo
    # espacio.
    # https://docs.python.org/3/library/re.html
    # https://platzi.com/blog/expresiones-regulares-python/
    pattern = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
    repl = " "
    new_content = ' '.join(re.sub(pattern, repl, new_content).split())

    return new_content


def remove_stopwords(content_list):
    """
    Remueve de una lista, las stopwords.
    """
    important_words = []
    for word in content_list:
        if word not in spanish_stopwords:
            important_words.append(word)

    return important_words

def delete_tildes(content):
    """
    Elimina las tildes de una cadena.
    """
    # http://guimi.net
    # Cambiamos caracteres modificados (áüç...) por los caracteres base (auc..)
    # Basado en una función de Miguel en
    # http://www.leccionespracticas.com/uncategorized/eliminar-tildes-con-python-solucionado/
    new_list = []
    for c in unicodedata.normalize('NFD', content):
        if unicodedata.category(c) != 'Mn':
            new_list += [c]

    s = ''.join(new_list)

    return s


# my_content = "hóla mundó ¿Cómo está la people?"
# print(delete_tildes(my_content))

class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

def tweet_stemming(content_list):
    """
    Realiza el stem de una lista de palabras.
    """
    result = []
    for word in content_list:
        result.append(stemmer.stem(word))

    return result
