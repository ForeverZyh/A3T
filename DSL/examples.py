import numpy as np

from DSL.transformations import Alphabet, REGEX, Transformation, INS, tUnion, SUB, DEL, Composition, Union

Alphabet.set_word_model()
dict_map = {}
word_lists = ["a", "cat", "plays", "with", "dog", "some", "cats", "dogs", "play", "in", "room", "the", "today"]
for i, s in enumerate(word_lists):
    dict_map[s] = i
Alphabet.set_alphabet(dict_map, np.random.normal(0, 1, (len(dict_map), 64)))
keep_same = REGEX(r'.*')
single2plural = Transformation(keep_same, DEL(lambda w: w == "a"),
                               SUB(lambda w: w in ["cat", "dog"], lambda w: w + "s"), keep_same)
plural2single = Transformation(keep_same, INS(lambda w: w == "a"),
                               SUB(lambda w: w in ["cats", "dogs"], lambda w: w[:-1]), keep_same)
verbsingle = Transformation(REGEX(r".*(dog|cat)"),
                            tUnion(SUB(lambda w: w == "play", lambda w: w + "s"), REGEX(r"plays")), keep_same)
verbplural = Transformation(REGEX(r".*(dog|cat)s"),
                            tUnion(SUB(lambda w: w == "plays", lambda w: w[:-1]), REGEX(r"play")), keep_same)
dogcat = Transformation(keep_same, SUB(lambda w: w in ["cat", "dog", "cats", "dogs"],
                                       lambda w: "dog" + w[3:] if w[:3] == "cat" else "cat" + w[3:]), keep_same)
untouched = Transformation(keep_same)
sub_b = Union(single2plural, plural2single, untouched)
b = Composition(Union(verbplural, verbsingle), sub_b, sub_b)

res = b.exact_space(("a", "cat", "plays", "with", "a", "dog"))
print(res)

sub_b1 = Union(dogcat, dogcat, untouched)
b1 = Composition(sub_b1, sub_b1, b)

Alphabet.set_char_model()
dict_map = {" ": 0}
for i in range(26):
    dict_map[chr(97 + i)] = i + 1
Alphabet.set_alphabet(dict_map, np.random.normal(0, 1, (len(dict_map), 2)))
keep_same = REGEX(r'.*')
untouched = Transformation(keep_same)
t1 = Transformation(keep_same, SUB(lambda c: c == 'a', lambda c: 'b'), keep_same)  # substitute 'a' with 'b'
unit = Union(t1, untouched)
a = Composition(unit, unit, unit)  # untouched or substitute with deletion
res = a.interval_space(tuple([(t,) for t in "abbaabba"]))
print(res)
