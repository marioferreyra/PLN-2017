"""
Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle

from languagemodeling.ngram import NGramGenerator


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargamos las opciones ingresadas
    model_file = str(opts['-i'])
    num_sents = int(opts['-n'])

    # Abrimo el Archivo que contiene el Modelo del lenguaje
    f = open(model_file, "rb")

    # Reconstruimos el objeto desde la representacion en cadena de bytes
    modelo = pickle.load(f)

    # Instanciamos un objeto NGramGenerator con el modelo obtenido
    generador = NGramGenerator(modelo)

    # Generamos un total de "num_sents" oraciones
    for _ in range(num_sents):
        sent = generador.generate_sent()
        # Unimos todos los tokens, pero separados por un espacio
        sent = " ".join(sent)
        print(sent)

    f.close()
