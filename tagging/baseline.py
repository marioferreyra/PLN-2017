from collections import defaultdict

class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        dict_word_tag_count = defaultdict(lambda : defaultdict(int))

        for sent in tagged_sents:
            for word, tag in sent:
                dict_word_tag_count[word][tag] += 1

        self.list_tag_count = []

        self.dict_word_tag = defaultdict(str)
        for word, tag_count in dict_word_tag_count.items():
            list_tag_count = list(tag_count.items())
            self.dict_word_tag[word] = list_tag_count[0][0]
            self.list_tag_count += list(tag_count.items())

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
        tag = "nc0s000"
        if not self.unknown(w):
            tag = self.dict_word_tag[w]

        return tag

    def unknown(self, w):
        """
        Check if a word is unknown for the model.

        w -- the word.
        """
        return (w not in self.dict_word_tag)


tagged_sents = [
            list(zip('el gato come pescado .'.split(),
                 'D N V N P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]
b = BaselineTagger(tagged_sents)