import numpy as np

from utils import tuple_set_union


class Alphabet:
    alphabet = []
    escaped_char = ""
    is_char_model = None
    mapping = None
    embedding = None
    max_len = None
    padding = None
    lines = open("./dataset/en.key").readlines()
    adjacent_keys = {}
    for line in lines:
        tmp = line.strip().split()
        adjacent_keys[tmp[0]] = tmp[1:]

    @staticmethod
    def partial_to_loss(x, y):
        return np.random.uniform(-1, 1, x.shape)

    '''
    :param alphabet: the alphabet of input strings, if it is char-level model, the type is str. If word-level model, 
        the type if tuple(str)
    :param escaped_char: the escaped_char that concatenate words. It is computed automatically, but also can be 
        specified
    :param is_char_model: a boolean value to indicate which model is. changed by set_word_model() and set_char_model()
    :param mapping: the mapping from char/word --> id
    :param embedding: the embedding of char/word
    :param max_len: the max_len of input string.
    :param padding: the padding char/word
    :param partial_to_loss: the partial derivative function that takes as inputs the input string s, and output the 
        partial derivatives.
    '''

    @staticmethod
    def toids(s):
        if Alphabet.max_len is None or Alphabet.padding is None:
            raise AttributeError("max_len or padding is not set!")

        ret = []
        for ss in s:
            ret.append(Alphabet.mapping[ss])
        for _ in range(Alphabet.max_len - len(s)):
            ret.append(Alphabet.mapping[Alphabet.padding])
        return np.array(ret)

    @staticmethod
    def set_alphabet(a: dict, embedding, specified_escaped_char=None):
        Alphabet.alphabet = [None] * len(a)
        Alphabet.mapping = a
        for s in a:
            Alphabet.alphabet[a[s]] = s
        Alphabet.embedding = embedding
        assert len(embedding) == len(a)

        if specified_escaped_char is not None:
            Alphabet.escaped_char = specified_escaped_char
        else:
            Alphabet.escaped_char = ""
            for i in range(10):
                Alphabet.escaped_char += r"\$"
                not_exist = True
                for s in Alphabet.alphabet:
                    if Alphabet.escaped_char in s:
                        not_exist = False
                        break
                if not_exist:
                    return

            raise AssertionError("cannot find an escaped_char!")

    @staticmethod
    def get_acc_alphabet(phi):
        alphabet_acc = []
        for c in Alphabet.alphabet:
            if phi(c):
                alphabet_acc.append(c)
        return alphabet_acc

    @staticmethod
    def set_word_model():
        Alphabet.is_char_model = False

    @staticmethod
    def set_char_model():
        Alphabet.is_char_model = True

    @staticmethod
    def to_interval_space(exact_space):
        ret = None
        for s in exact_space:
            ret = tuple_set_union(ret, tuple([(t,) for t in s]))
        return ret

    @staticmethod
    def to_convex_hull(exact_space, orgin_input):
        ret = Alphabet.to_interval_space(exact_space)
        Max_modify = 0
        for s in exact_space:
            modify = abs(len(s) - len(orgin_input))
            min_len = min(len(s), len(orgin_input))
            for i in range(min_len):
                if s[i] != orgin_input[i]:
                    modify += 1
            Max_modify = max(Max_modify, modify)
        return ret, Max_modify
