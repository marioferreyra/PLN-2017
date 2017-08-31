from collections import defaultdict


class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        # Diccionario = word : {tag : cantidad}
        dict_word_tag_count = defaultdict(lambda: defaultdict(int))
        for sent in tagged_sents:
            for word, tag in sent:
                dict_word_tag_count[word][tag] += 1

        self.dict_word_tag = defaultdict(str)  # Diccionario = word : tag

        for word, tag_count in dict_word_tag_count.items():
            # Maximo de la lista (tag, cantidad)
            tag_most_frec = max(tag_count.items(), key=lambda x: x[1])[0]

            self.dict_word_tag[word] = tag_most_frec

    def tag(self, sent):
        """
        Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """
        Tag a word.

        w -- the word.
        """
        tag = "nc0s000"  # Supongamos que la palabra es desconocida

        # Si no es desconocida, devolvemos su etiqueta mas frecuente
        if not self.unknown(w):
            tag = self.dict_word_tag[w]

        return tag

    def unknown(self, w):
        """
        Check if a word is unknown for the model.

        w -- the word.
        """
        # Si la palabra no esta en el diccionario ==> True
        # Si no ==> False
        return (w not in self.dict_word_tag)
