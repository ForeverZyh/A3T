import numpy as np
import keras
import keras.backend as K
import heapq
import copy


class Gradient(keras.layers.Layer):
    def __init__(self, y, **kwargs):
        super(Gradient, self).__init__(**kwargs)
        self.y = y

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Gradient, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        return K.gradients(self.y, x)[0]

    def compute_output_shape(self, input_shape):
        return input_shape


def tuple_set_union(ret0: tuple, ret1: tuple):
    if ret0 is None:
        return ret1
    if ret1 is None:
        return ret0
    ret = ()
    max_len = max(len(ret0), len(ret1))
    for i in range(max_len):
        if i >= len(ret0):
            r0 = [""]
        else:
            r0 = ret0[i]
        if i >= len(ret1):
            r1 = [""]
        else:
            r1 = ret1[i]
        ret += (tuple(set(r0).union(set(r1))),)

    return ret


class Beam:
    def __init__(self, budget):
        '''
        The queue are contain two elements: (data, score), while the score is the ranking key (descending)
        :param budget: the beam search budget
        '''
        self.budget = budget
        self.queue = []

    def add(self, data, score):
        if len(self.queue) == self.budget:
            a, b = heapq.heappop(self.queue)
            if a > score:
                socre, data = a, b
        heapq.heappush(self.queue, (score, data))

    def extend(self, others):
        if isinstance(others, list):
            for data, score in others:
                self.add(data, score)
        else:
            for data, score in others.queue:
                self.add(data, score)

    def check_balance(self):
        ret = []
        while len(self.queue) > 0:
            a, b = heapq.heappop(self.queue)
            ret.append([b, a])
        return ret

    def is_same(self, others: list):
        if len(others) != len(self.queue):
            return False
        others.sort(key=lambda x: -x[1])
        q = copy.deepcopy(self.queue)
        for i in range(len(others)):
            a, b = heapq.heappop(q)
            if others[i][0] != b or others[i][1] != a:
                return False

        return True


class Dict:
    def __init__(self, char2id):
        self.char2id = char2id
        self.id2char = [" "] * len(char2id)
        for c in char2id:
            self.id2char[char2id[c]] = c

    def to_string(self, ids):
        return "".join([self.id2char[x] for x in ids])

    def to_ids(self, s):
        return np.array([self.char2id[c] for c in s])
