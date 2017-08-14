import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descargar las stopwords usando el comando nltk.download()
spanish_stopwords = stopwords.words('spanish')  # Stopwords del español


def delete_tildes(content):
    """
    Remplazamos todos los caracteres de una cadena por su version sin acentuar
    (sin tilde).

    Ejemplo:
        >>> s = "hóla mùndó, ¿Cómo está la people?"
        >>> delete_tildes(s)
        >>> hola mundo, ¿Como esta la people?
    """
    new_list = []
    for c in unicodedata.normalize('NFD', content):
        if unicodedata.category(c) != 'Mn':
            new_list += [c]

    return ''.join(new_list)


def tweet_cleaner(content):
    """
    Limpiamos el contenido de un tweet de la siguiente forma:
        * Convertimos el texto a minuscula.
        * Removemos URL.
        * Removemos signos de puntuacion.
        * Removemos nombres de usuarios (ie. @user).
        * Removemos caracteres no alfanumericos.

    Finalmente retornamos una cadena, es decir, el tweet limpio.
    """
    pattern = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
    repl = " "
    new_content = ' '.join(re.sub(pattern, repl, content.lower()).split())

    return new_content


def remove_repeated(word):
    """
    Remueve de una palabra los caracteres repetidos.
    """
    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'

    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:
        return remove_repeated(repl_word)
    else:
        return repl_word


def reduce_laught_expression(word):
    """
    Busca patrones repetidos de una palabra y devuelve la parte que se repite,
    si no hay ninguna parte que se repita devuelve la palabra que se recibio de
    entrada.
    """
    regex = r"(\b([a-z]{2,}?)\2+\b)"
    pattern = re.findall(regex, word, re.X | re.I)

    if pattern != []:
        return pattern[0][1]

    return word


def change_to_risas(word):
    """
    Las expresiones que denotan risas por ejemplo "jajaja", "jejeje" entre
    otras son reemplazadas por la palabra "risas".
    """
    # Convertimos la expresion de risas a su "menor" forma:
    # Por ejemplo: "jajajajaja" --> "ja"
    reduce_laught = reduce_laught_expression(word)

    # Si la expresion esta en el conjunto de expresiones de risas, reemplazamos
    # la palabra por "risas" sino devolvemos la palabra que se recibio de
    # entrada
    laught_expression = {"ja", "je", "ji", "jo", "ju", "ah", "ha", "hu"}
    if reduce_laught in laught_expression:
        return "risas"
    else:
        return reduce_laught


def remove_stopwords(content_list):
    """
    Removemos de una lista de palabras, aquellas que son muy frecuentes
    pero que no aportan gran valor sintáctico, es decir las stopwords.
    """
    important_words = []
    for word in content_list:
        if word not in spanish_stopwords:
            important_words.append(word)

    return important_words


def tweet_stemming(content_list):
    """

    Dada una lista de palabras, a cada una de ellas se le realiza el Stemming.
    Stemming es el proceso por el cual transformamos cada palabra en su raiz.
    Por ejemplo:
        maravilloso     |
        maravilla       |-> maravill
        maravillarse    |
    """
    stemmer = SnowballStemmer("spanish")  # Lematizador del español
    result = []
    for word in content_list:
        result.append(stemmer.stem(word))

    return result
