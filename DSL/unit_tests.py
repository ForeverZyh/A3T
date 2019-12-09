import numpy as np
import time
import signal

from DSL.transformations import Alphabet, REGEX, Transformation, INS, tUnion, SWAP, SUB, DEL, Composition, Union, \
    tuple_set_union

Alphabet.set_char_model()
dict_map = {" ": 0}
for i in range(26):
    dict_map[chr(97 + i)] = i + 1
Alphabet.set_alphabet(dict_map, np.random.normal(0, 1, (len(dict_map), 2)))
regex1 = REGEX(r'.*')
regex2 = REGEX(r'a')
t1 = Transformation(regex1, INS(lambda c: c == 'a'), regex1)  # insert 'a' at any place
t2 = Transformation(regex1, tUnion(SUB(lambda c: c == 'a', lambda c: 'b'), SUB(lambda c: c == 'b', lambda c: 'a')),
                    regex1)  # substitute 'a' with 'b' or 'b' with 'a'
t3 = Transformation(regex2, SWAP(lambda c: True, lambda c: True), regex1)  # swap two adjacent after a leading 'a'
t4 = Transformation(regex1, DEL(lambda c: c == 'b'), regex1)  # delete 'b' at any place
a = Composition(Union(t1, t2), t3, t4)  # first delete then swap then (insert or substitute)


def char_test():
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


def random_generator_200s():
    s = ""
    for i in range(200):
        s += chr(np.random.randint(0, 26) + 97)
        if np.random.rand() < 0.2:
            s += " "
    return s


def is_precise(res, precise_res):
    merged_res = tuple_set_union(precise_res, res)
    assert len(res) == len(merged_res)
    for x, y in zip(res, merged_res):
        assert len(set(x).symmetric_difference(set(y))) == 0

    for x, y in zip(precise_res, merged_res):
        if len(set(x).symmetric_difference(set(y))) > 0:
            return False
    return True


def interval_char_test():
    ss = ['abcsabb'] + [random_generator_200s() for _ in range(0)]
    for s in ss:
        res = a.interval_space(tuple([(t,) for t in s]))
        precise_res = Alphabet.to_interval_space(a.exact_space(s))
        if is_precise(res, precise_res):
            print("Precise for input: " + s)
        else:
            print("Imprecise for input: " + s)

    print("interval_char_test passed...")


def convex_char_test():
    ss = ['abcsabb'] + [random_generator_200s() for _ in range(0)]
    for s in ss:
        precise_res = a.exact_space(s)
        convex = Alphabet.to_convex_hull(precise_res, s)
        print(convex[1])

    print("convex_char_test passed...")


def throughput_test():
    t = time.process_time()
    for _ in range(5):
        s = "a" + random_generator_200s()
        a.exact_space(s)

    print("throughput_test using " + str(time.process_time() - t) + "(s) time ...")


char_test()
interval_char_test()
convex_char_test()
throughput_test()

Alphabet.set_word_model()
dict_map = {}
word_lists = ["a", "cat", "plays", "with", "dog", "some", "cats", "dogs", "play", "in", "room", "the", "today"]
for i, s in enumerate(word_lists):
    dict_map[s] = i
Alphabet.set_alphabet(dict_map, np.random.normal(0, 1, (len(dict_map), 64)))
single2plural = Transformation(regex1, DEL(lambda w: w == "a"),
                               SUB(lambda w: w in ["cat", "dog"], lambda w: w + "s"), regex1)
plural2single = Transformation(regex1, INS(lambda w: w == "a"),
                               SUB(lambda w: w in ["cats", "dogs"], lambda w: w[:-1]), regex1)
verbsingle = Transformation(REGEX(r".*(dog|cat)"),
                            tUnion(SUB(lambda w: w == "play", lambda w: w + "s"), REGEX(r"plays")), regex1)
verbplural = Transformation(REGEX(r".*(dog|cat)s"),
                            tUnion(SUB(lambda w: w == "plays", lambda w: w[:-1]), REGEX(r"play")), regex1)
dogcat = Transformation(regex1, SUB(lambda w: w in ["cat", "dog", "cats", "dogs"],
                                    lambda w: "dog" + w[3:] if w[:3] == "cat" else "cat" + w[3:]), regex1)
same = Transformation(regex1)
sub_b = Union(single2plural, plural2single, same)
b = Composition(Union(verbplural, verbsingle), sub_b, sub_b)
sub_b1 = Union(dogcat, dogcat, same)
b1 = Composition(sub_b1, sub_b1, b)


def word_test():
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


def word_interval_test():
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

        precise_res = Alphabet.to_interval_space(x.exact_space(s_))
        print(res)
        print(precise_res)
        if is_precise(res, precise_res):
            print("Precise for input: " + str(s))
        else:
            print("Imprecise for input: " + str(s))

    print("interval_word_test passed...")


word_test()
word_interval_test()
