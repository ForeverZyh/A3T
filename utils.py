import numpy as np
import heapq

inf = 1e10


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
        '''
        add (data, score) into queue
        :param data: candidate
        :param score: score of the candidate
        :return: True if added in, False otherwise
        '''
        if data in self.in_queue:  # if data is ready in the priority queue, we update the score in self.in_queue
            if self.in_queue[data] < score:
                self.in_queue[data] = score
                return True
            return False

        ret = True
        if len(self.queue) == self.budget:  # if size(queue) == budget, we need to remove one
            while True:
                a, b = heapq.heappop(self.queue)
                # the top of the priority queue may not be smallest, because it may contain new value in self.in_queue
                if a == self.in_queue[b]:  # if the value is not updated, then it is smallest
                    break  # remove (a, b)
                else:
                    heapq.heappush(self.queue,
                                   (self.in_queue[b], b))  # otherwise, update in the priority queue (lazily)
            del self.in_queue[b]  # remove (a, b) from self.in_queue
            if a > score:  # if the old (a, b) is better then new (score, data), we replace (score, data) with (a, b)
                score, data = a, b
                ret = False

        heapq.heappush(self.queue, (score, data))  # add (score, data)
        self.in_queue[data] = score  # update in self.in_queue
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
            ret.append((data, self.in_queue[data]))
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


def swap_pytorch(x, p1, p2):
    z = x[p1].clone()
    x[p1] = x[p2]
    x[p2] = z
