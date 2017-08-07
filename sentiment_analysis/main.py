from read_xml import readXMLTrain
from preprocessing import tweet_cleaner, remove_stopwords, delete_tildes
from preprocessing import RepeatReplacer

from nltk.tokenize import TweetTokenizer

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


tweets = readXMLTrain("/home/mario/Escritorio/TEST/entrenador.xml")
# Solo hay 3 tweets en este XML (rango 0, 1, 2)
tweet = tweets[2]


print("Tweet")
print("=======")

print("Contenido")
print("=========")
my_content = tweet.content
print(my_content)
if tweet.polarity == 0:
    print("NONE")
elif tweet.polarity == 1:
    print("NEGATIVO")
elif tweet.polarity == 2:
    print("NEUTRO")
elif tweet.polarity == 3:
    print("POSITIVO")

print("\nContenido, con las tildes eliminadas")
print("====================================")
my_content = delete_tildes(my_content)
print(my_content)

print("\nContenido, limpiado")
print("===================")
my_content = tweet_cleaner(my_content)
print(my_content)

print("\nContenido Tokenizando")
print("=====================")
my_content = TweetTokenizer().tokenize(my_content)
print(my_content)

print("\nContenido, removiendo Stopwords")
print("===============================")
my_content = remove_stopwords(my_content)
print(my_content)
