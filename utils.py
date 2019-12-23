import numpy as np
import keras
import keras.backend as K
import heapq
import copy
from multiprocessing import Pool, Process, SimpleQueue, Pipe


class Multiprocessing:
    @staticmethod
    def work(fun, child_conn, args):
        ret = fun(args[0], child_conn, args[2])
        child_conn.send(("close", ret[0]))

    @staticmethod
    def mapping(fun, args_list, processes, partial_to_loss):
        ans = [None] * len(args_list)
        pipes = []
        for batch_start in range(0, len(args_list), processes):
            ps = []
            for i in range(batch_start, min(batch_start + processes, len(args_list))):
                parent_conn, child_conn = Pipe()
                pipes.append(parent_conn)
                p = Process(target=Multiprocessing.work, args=(fun, child_conn, args_list[i]))
                p.start()
                ps.append(p)

            unfinished = len(ps)
            while unfinished > 0:
                for i in range(batch_start, min(batch_start + processes, len(args_list))):
                    if pipes[i] is not None:
                        s = pipes[i].recv()
                        if len(s) == 2 and s[0] == "close":
                            ans[i] = s[1]
                            pipes[i] = None
                            unfinished -= 1
                        else:
                            pipes[i].send(partial_to_loss(s, args_list[i][1]))

            for p in ps:
                p.join()

        return ans


class MultiprocessingWithoutPipe:
    @staticmethod
    def work(fun, num, q, args):
        np.random.seed(num)
        ret = fun(*args)
        q.put((num, ret))

    @staticmethod
    def mapping(fun, args_list, processes):
        ans = [None] * len(args_list)
        q = SimpleQueue()
        for batch_start in range(0, len(args_list), processes):
            ps = []
            for i in range(batch_start, min(batch_start + processes, len(args_list))):
                p = Process(target=MultiprocessingWithoutPipe.work, args=(fun, i, q, args_list[i]))
                p.start()
                ps.append(p)

            while not q.empty():
                num, ret = q.get()
                ans[num] = ret

            for p in ps:
                p.join()

        while not q.empty():
            num, ret = q.get()
            ans[num] = ret

        return ans


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
        self.in_queue = {}

    def add(self, data, score):
        if data in self.in_queue:
            if self.in_queue[data] < score:
                self.in_queue[data] = score
                return True
            return False

        ret = True
        if len(self.queue) == self.budget:
            while True:
                a, b = heapq.heappop(self.queue)
                if a == self.in_queue[b]:
                    break
                else:
                    heapq.heappush(self.queue, (self.in_queue[b], b))
            del self.in_queue[b]
            if a > score:
                score, data = a, b
                ret = False

        heapq.heappush(self.queue, (score, data))
        self.in_queue[data] = score
        return ret

    def extend(self, others):
        if isinstance(others, list):
            for data, score in others:
                self.add(data, score)
        else:
            assert False
            # for data, score in others.queue:
            #     self.add(data, score)

    def check_balance(self):
        ret = []
        for data in self.in_queue:
            ret.append([data, self.in_queue[data]])
        ret.sort(key=lambda x: -x[1])
        return ret

    def is_same(self, others: list):
        if len(others) != len(self.queue):
            return False
        others.sort(key=lambda x: -x[1])
        for i in range(len(others)):
            data, score = others[i]
            if data not in self.in_queue or self.in_queue[data] != score:
                return False

        return True


class UnorderedBeam:
    def __init__(self, budget):
        self.budget = budget
        self.queue = []

    def add(self, data):
        self.queue.append(data)

    def extend(self, others):
        if isinstance(others, list):
            self.queue.extend(others)
        else:
            assert False
            # for data, score in others.queue:
            #     self.add(data, score)

    def check_balance(self):
        ids = np.random.randint(0, len(self.queue), self.budget)
        ret = []
        for id in ids:
            ret.append(self.queue[id])
        return ret


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
