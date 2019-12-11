import numpy as np


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
        for i, (data_in_queue, score_in_queue) in enumerate(self.queue):
            if data_in_queue == data:
                self.queue[i][1] = max(self.queue[i][1], score)
                return

        self.queue.append([data, score])

    def extend(self, others):
        if isinstance(others, list):
            for data, score in others:
                self.add(data, score)
        else:
            for data, score in others.queue:
                self.add(data, score)

    def check_balance(self):
        if self.budget < len(self.queue):
            self.queue.sort(key=lambda x: -x[1])
            self.queue = self.queue[:self.budget]
        return self.queue

    def add_score(self, score):
        for i in range(len(self.queue)):
            self.queue[i][1] += score

    def is_same(self, others: list):
        if len(others) != len(self.queue):
            return False
        others.sort(key=lambda x: -x[1])
        self.queue.sort(key=lambda x: -x[1])
        for i in range(len(others)):
            if others[i][0] != self.queue[i][0] or others[i][1] != self.queue[i][1]:
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
