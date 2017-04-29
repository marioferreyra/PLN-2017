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
            # Lista (tags, cantidad) ordenado de mayor a menor
            list_tags_count_sorted = sorted(tag_count.items(),
                                            key=lambda x: x[1],
                                            reverse=True)

            # Lista de tags
            list_tags = [tag for tag, _ in list_tags_count_sorted]

            self.dict_word_tag[word] = list_tags[0]  # Tag mas frecuente

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


tagged_sents = [
            list(zip('el gato come pescado gato gato .'.split(),
                 'D N V N V V P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]
b = BaselineTagger(tagged_sents)
