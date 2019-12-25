import numpy as np
import time
import signal

from DSL.transformations import REGEX, Transformation, INS, tUnion, SWAP, SUB, DEL, Composition, Union
from DSL.Alphabet import Alphabet
from utils import tuple_set_union, Beam, Multiprocessing, MultiprocessingWithoutPipe


def do_test():
    Alphabet.set_char_model()
    Alphabet.max_len = 10
    Alphabet.padding = " "
    dict_map = {" ": 0}
    for i in range(26):
        dict_map[chr(97 + i)] = i + 1
    Alphabet.set_alphabet(dict_map, np.random.normal(0, 1, (len(dict_map), 2)))
    regex1 = REGEX(r'.*')
    regex2 = REGEX(r'a')
    untouched = Transformation(regex1)
    t1 = Transformation(regex1, INS(lambda c: c == 'a'), regex1)  # insert 'a' at any place
    t2 = Transformation(regex1,
                        tUnion(SUB(lambda c: c == 'a', lambda c: {'b'}), SUB(lambda c: c == 'b', lambda c: {'a'})),
                        regex1)  # substitute 'a' with 'b' or 'b' with 'a'
    t3 = Transformation(regex2, SWAP(lambda c: True, lambda c: True), regex1)  # swap two adjacent after a leading 'a'
    t4 = Transformation(regex1, DEL(lambda c: c == 'b'), regex1)  # delete 'b' at any place
    t12 = Union(t1, t2)
    ssub = SUB(lambda c: c in Alphabet.adjacent_keys, lambda c: Alphabet.adjacent_keys[c])
    sub_single = Transformation(regex1, ssub, regex1)
    dl_sub = Composition(sub_single, sub_single, sub_single)
    a = Composition(t12, t3, t4)  # first delete then swap then (insert or substitute)
    t12_untouched = Union(t12, untouched)
    t3_untouched = Union(t3, untouched)
    t4_untouched = Union(t4, untouched)
    a_untouched = Composition(t12_untouched, t3_untouched, t4_untouched)

    def get_max_delta_test0():
        assert t1.get_max_delta() == [1, 1]
        assert t2.get_max_delta() == [0, 1]
        assert t3.get_max_delta() == [0, 2]
        assert t4.get_max_delta() == [1, 0]
        assert untouched.get_max_delta() == [0, 0]
        assert t12.get_max_delta() == [1, 1]
        assert sub_single.get_max_delta() == [0, 1]
        assert dl_sub.get_max_delta() == [0, 3]
        assert a.get_max_delta() == [1, 0]
        assert t12_untouched.get_max_delta() == [1, 1]
        assert t3_untouched.get_max_delta() == [0, 2]
        assert t4.get_max_delta() == [1, 0]
        assert a_untouched.get_max_delta() == [1, 0]
        print("get_max_delta_test0 passed...")

    get_max_delta_test0()

    # random_sample_test(a_untouched)
    # char_test(a)
    interval_convex_char_test(a, dl_sub)
    throughput_test(a)
    throughput_test1(dl_sub)
    beam_search_adversarial_test(a_untouched, t12_untouched, t3_untouched, t4_untouched)

    Alphabet.set_word_model()
    dict_map = {}
    word_lists = ["a", "cat", "plays", "with", "dog", "some", "cats", "dogs", "play", "in", "room", "the", "today", " "]
    for i, s in enumerate(word_lists):
        dict_map[s] = i
    Alphabet.set_alphabet(dict_map, np.random.normal(0, 1, (len(dict_map), 64)))
    single2plural = Transformation(regex1, DEL(lambda w: w == "a"),
                                   SUB(lambda w: w in ["cat", "dog"], lambda w: {w + "s"}), regex1)
    plural2single = Transformation(regex1, INS(lambda w: w == "a"),
                                   SUB(lambda w: w in ["cats", "dogs"], lambda w: {w[:-1]}), regex1)
    verbsingle = Transformation(REGEX(r".*(dog|cat)"),
                                tUnion(SUB(lambda w: w == "play", lambda w: {w + "s"}), REGEX(r"plays")), regex1)
    verbplural = Transformation(REGEX(r".*(dog|cat)s"),
                                tUnion(SUB(lambda w: w == "plays", lambda w: {w[:-1]}), REGEX(r"play")), regex1)
    dogcat = Transformation(regex1, SUB(lambda w: w in ["cat", "dog", "cats", "dogs"],
                                        lambda w: {"dog" + w[3:]} if w[:3] == "cat" else {"cat" + w[3:]}), regex1)
    same = Transformation(regex1)
    sub_b = Union(single2plural, plural2single, same)
    b = Composition(Union(verbplural, verbsingle), sub_b, sub_b)
    sub_b1 = Union(dogcat, same)
    b1 = Composition(sub_b1, sub_b1, b)

    def get_max_delta_test1():
        assert single2plural.get_max_delta() == [1, 0]
        assert plural2single.get_max_delta() == [1, 1]
        assert verbsingle.get_max_delta() == [0, 1]
        assert verbplural.get_max_delta() == [0, 1]
        assert dogcat.get_max_delta() == [0, 1]
        assert same.get_max_delta() == [0, 0]
        assert sub_b.get_max_delta() == [1, 1]
        assert b.get_max_delta() == [1, 2]
        assert sub_b1.get_max_delta() == [0, 1]
        assert b1.get_max_delta() == [1, 2]
        print("get_max_delta_test1 passed...")

    get_max_delta_test1()
    word_test(b, b1)
    word_interval_test(b, b1, dogcat)


def random_sample_test(a):
    res = set(a.exact_space("abcsabb"))
    res_dict = {}
    for s in res:
        res_dict[s] = len(res_dict)
    dist = np.zeros(len(res))
    times = 1000
    rets = MultiprocessingWithoutPipe.mapping(a.random_sample, [("abcsabb", 5) for _ in range(times)], 8)
    for random_res in rets:
        for s in random_res:
            assert s in res
            dist[res_dict[s]] += 1
    print(dist * 1.0 / times)
    print("random_sample_test passed...")


def char_test(a):
    res = a.exact_space("abcsabb")
    correct_res = set(
        "aascabb asacabb ascaabb ascabab ascabba aacbsab acabsab acbasab acbsaab acbsaba bscabb ascbbb ascaab ascaba bcbsab acasab acbsbb acbsaa".split())
    print(res)
    assert len(res.difference(correct_res)) == 0 and len(res) == len(correct_res)
    print("char_test passed...")
    '''
    acsabb abcsab
    ascabb acbsab
    aascabb asacabb ascaabb ascabab ascabba aacbsab acabsab acbasab acbsaab acbsaba bscabb ascbbb ascaab ascaba bcbsab acasab acbsbb acbsaa
    '''


def random_generator(length):
    s = ""
    for i in range(length):
        s += chr(np.random.randint(0, 26) + 97)
        if np.random.rand() < 0.2:
            s += " "
    return s


def is_precise(res, precise_res):
    merged_res = tuple_set_union(precise_res, res)
    if res is None:
        assert precise_res is None
        return True

    assert len(res) == len(merged_res)
    for x, y in zip(res, merged_res):
        assert len(set(x).symmetric_difference(set(y))) == 0

    for x, y in zip(precise_res, merged_res):
        if len(set(x).symmetric_difference(set(y))) > 0:
            return False
    return True


def interval_convex_char_test(a, dl_sub):
    ss = ['abcsabb'] + ["a" + random_generator(20) for _ in range(5)]
    for s in ss:
        res = a.interval_space(tuple([(t,) for t in s]))
        max_delta = a.get_max_delta()
        compute_delta = max_delta[0] * len(s) + max_delta[1]
        precise_res, precise_delta = Alphabet.to_convex_hull(a.exact_space(s), s)
        assert compute_delta >= precise_delta
        if is_precise(res, precise_res) and compute_delta == precise_delta:
            print("Precise for input: " + s)
        else:
            print(res, compute_delta)
            print(precise_res, precise_delta)
            print("Imprecise for input: " + s)

    ss = [random_generator(10) for _ in range(5)]
    for s in ss:
        res = dl_sub.interval_space(tuple([(t,) for t in s]))
        max_delta = dl_sub.get_max_delta()
        compute_delta = max_delta[0] * len(s) + max_delta[1]
        precise_res, precise_delta = Alphabet.to_convex_hull(dl_sub.exact_space(s), s)
        # substitution should be precise
        assert is_precise(res, precise_res) and compute_delta == precise_delta

    print("interval_convex_char_test passed...")


def throughput_test1(dl_sub):
    t = time.time()
    ans = Multiprocessing.mapping(dl_sub.beam_search_adversarial,
                                  [(random_generator(300), None, 3) for _ in range(16)], 8, Alphabet.partial_to_loss)
    print("throughput_test1 using " + str(time.time() - t) + "(s) time ...")


def throughput_test(a):
    t = time.process_time()
    for _ in range(5):
        s = "a" + random_generator(300)
        a.exact_space(s)

    print("throughput_test using " + str(time.process_time() - t) + "(s) time ...")


class SimpleModel:
    def __init__(self):
        import tensorflow as tf
        self.embedding_dim = 2
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, Alphabet.max_len])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        embed = tf.Variable(tf.random_normal([27, self.embedding_dim]))
        x = tf.gather(embed, self.x)
        W = tf.Variable(tf.random_normal([Alphabet.max_len * self.embedding_dim, 3]))
        b = tf.Variable(tf.random_normal([3]))
        self.loss = tf.reduce_sum(
            (tf.matmul(tf.reshape(x, (-1, Alphabet.max_len * self.embedding_dim)), W) + b - self.y) ** 2, axis=-1)
        self.partial_loss = tf.gradients(self.loss, x)[0]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def partial_to_loss(self, embedding, y):
        return self.sess.run(self.partial_loss,
                             feed_dict={self.x: np.expand_dims(embedding, axis=0), self.y: np.expand_dims(y, axis=0)})[
            0]


def beam_search_adversarial_test(a, t12, t3, t4):
    s = 'abcsabb'
    model = SimpleModel()
    Alphabet.partial_to_loss = model.partial_to_loss
    budget = 4
    y = np.array([0, 1, 0])
    beams = a.beam_search_adversarial(s, y, budget)
    print(beams)
    ans = Multiprocessing.mapping(a.beam_search_adversarial,
                                  [(s, y, budget) for _ in range(16)], 8, Alphabet.partial_to_loss)
    for aans in ans:
        assert tuple(aans) == tuple(beams[0])
    outputs = a.exact_space(s)
    for output, score in beams:
        assert output in outputs

    worse = -1e20
    worse_output = ""
    for output in outputs:
        t = model.sess.run(model.loss,
                           feed_dict={model.x: np.expand_dims(Alphabet.toids(output), axis=0),
                                      model.y: np.expand_dims(y, axis=0)})
        if worse < t:
            worse = t
            worse_output = output
        # print(output, t)

    find = False
    for output, score in beams:
        if output == worse_output:
            find = True
            print("Precise beam search!")
            break

    if not find:
        print("Imprecise beam search!")
    print("beam_search_adversarial_test passed...")


def word_test(b, b1):
    res = b.exact_space(("a", "cat", "plays", "with", "a", "dog"))
    correct_res = {("a", "cat", "plays", "with", "a", "dog"), ("cats", "play", "with", "a", "dog"),
                   ("a", "cat", "plays", "with", "dogs"), ("cats", "play", "with", "dogs")}
    assert len(res.difference(correct_res)) == 0 and len(res) == len(correct_res)

    res = b.exact_space(("today", "a", "cat", "plays", "with", "dogs", "in", "the", "room"))
    correct_res1 = {("today",) + x + ("in", "the", "room") for x in correct_res}
    assert len(res.difference(correct_res1)) == 0 and len(res) == len(correct_res1)

    res = b1.exact_space(("a", "cat", "plays", "with", "a", "dog"))
    correct_res = {("a", "cat", "plays", "with", "a", "dog"), ("cats", "play", "with", "a", "dog"),
                   ("a", "cat", "plays", "with", "dogs"), ("cats", "play", "with", "dogs"),
                   ("a", "dog", "plays", "with", "a", "dog"), ("dogs", "play", "with", "a", "dog"),
                   ("a", "dog", "plays", "with", "dogs"), ("dogs", "play", "with", "dogs"),
                   ("a", "cat", "plays", "with", "a", "cat"), ("cats", "play", "with", "a", "cat"),
                   ("a", "cat", "plays", "with", "cats"), ("cats", "play", "with", "cats"),
                   ("a", "dog", "plays", "with", "a", "cat"), ("dogs", "play", "with", "a", "cat"),
                   ("a", "dog", "plays", "with", "cats"), ("dogs", "play", "with", "cats")}
    assert len(res.difference(correct_res)) == 0 and len(res) == len(correct_res)

    res = b1.exact_space(("today", "a", "cat", "plays", "with", "dogs", "in", "the", "room"))
    correct_res1 = {("today",) + x + ("in", "the", "room") for x in correct_res}
    assert len(res.difference(correct_res1)) == 0 and len(res) == len(correct_res1)
    print("word_test passed...")


def word_interval_test(b, b1, dogcat):
    ss = [("a", "cat", "plays", "with", "a", "dog"),
          ("today", "a", "cat", "plays", "with", "dogs", "in", "the", "room"),
          ("a", "cat", "plays", "with", "a", "dog"),
          ("today", "a", "cat", "plays", "with", "dogs", "in", "the", "room")]

    bb = [b, b, b1, b1]

    for s_, x in zip(ss, bb):
        s = [(t,) for t in s_]

        def handler(signum, frame):
            raise Exception("Time out for input: ")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        try:
            res = x.interval_space(tuple(s))
        except Exception as exc:
            print(str(exc) + str(s))
            continue

        precise_res, precise_delta = Alphabet.to_convex_hull(x.exact_space(s_), s_)
        max_delta = x.get_max_delta()
        compute_delta = max_delta[0] * len(s) + max_delta[1]
        assert compute_delta >= precise_delta
        if is_precise(res, precise_res) and compute_delta == precise_delta:
            print("Precise for input: " + str(s))
        else:
            print(res, compute_delta)
            print(precise_res, precise_delta)
            print("Imprecise for input: " + str(s))

    for s_ in ss:
        s = [(t,) for t in s_]

        def handler(signum, frame):
            raise Exception("Time out for input: ")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        try:
            res = dogcat.interval_space(tuple(s))
        except Exception as exc:
            print(str(exc) + str(s))
            continue

        precise_res, precise_delta = Alphabet.to_convex_hull(dogcat.exact_space(s_), s_)
        max_delta = dogcat.get_max_delta()
        compute_delta = max_delta[0] * len(s) + max_delta[1]
        assert is_precise(res, precise_res) and compute_delta == precise_delta

    print("interval_word_test passed...")
