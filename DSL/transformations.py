import re
import itertools
import numpy as np

from utils import tuple_set_union, Beam
from DSL.Alphabet import Alphabet


class INS:
    def __init__(self, phi):
        self.phi = phi
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)

    def exact_space(self, s):
        ret = set()
        for c in self.alphabet_acc:
            if Alphabet.is_char_model:  # if character-level model
                ret.add((0, c))
            else:  # if word-level model
                ret.add((0, (c,)))
        return ret

    def interval_space(self, s):
        return {0: (tuple(self.alphabet_acc),)}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        assert b > 0
        ret = Beam(b)
        for c in self.alphabet_acc:
            if Alphabet.is_char_model:  # if character-level model
                new_output = output + c
            else:  # if word-level model
                new_output = output + (c,)
            end_pos = min(len(new_output) - 1, input_pos - 1)
            if end_pos == len(new_output) - 1:
                score = np.sum(
                    partial_loss[end_pos] * (Alphabet.mapping[new_output[end_pos]] - Alphabet.mapping[s[end_pos]]))
            else:
                score = 0
            ret.add(new_output, score)

        return {input_pos: ret.check_balance()}


class DEL:
    def __init__(self, phi):
        self.phi = phi
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)

    def exact_space(self, s):
        if len(s) > 0 and self.phi(s[0]):
            if Alphabet.is_char_model:  # if character-level model
                return {(1, "")}
            else:  # if word-level model
                return {(1, ())}
        else:
            return set()

    def interval_space(self, s):
        if len(s) > 0 and not set(self.alphabet_acc).isdisjoint(set(s[0])):
            return {1: ()}
        else:
            return {}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        assert b > 0
        if input_pos < len(s) and self.phi(s[input_pos]):
            end_pos = min(len(output) - 1, input_pos)
            if end_pos == input_pos:
                score = np.sum(
                    partial_loss[end_pos] * (Alphabet.mapping[output[end_pos]] - Alphabet.mapping[s[end_pos]]))
            else:
                score = 0
            return {input_pos + 1: [[output, score]]}
        else:
            return {}


class REGEX:
    def __init__(self, regex):
        self.regex = re.compile(regex)
        self.cache_exact = {}
        self.cache_interval = {}

    def exact_space(self, s):
        if s in self.cache_exact:
            return self.cache_exact[s]
        ret = set()
        for i in range(len(s) + 1):
            if Alphabet.is_char_model:  # if character-level model
                if self.regex.fullmatch(s[:i]):
                    ret.add((i, s[:i]))
            else:  # if word-level model
                if self.regex.fullmatch(Alphabet.escaped_char.join(s[:i])):
                    ret.add((i, s[:i]))
        self.cache_exact[s] = ret
        return ret

    def interval_space(self, s):
        # TODO figure out a more efficient way to do this. One optimization maybe use DFA so that some prefix can be eliminated at early stage. However, it is still exponential.
        if s in self.cache_interval:
            return self.cache_interval[s]
        ret = {}
        for i in range(len(s) + 1):
            for s_tuple in itertools.product(*(s[:i])):
                if (Alphabet.is_char_model and self.regex.fullmatch("".join(s_tuple))) or (
                        not Alphabet.is_char_model and self.regex.fullmatch(Alphabet.escaped_char.join(s_tuple))):
                    if i not in ret:
                        ret[i] = tuple([(t,) for t in s_tuple])
                    else:
                        ret[i] = tuple_set_union(ret[i], tuple([(t,) for t in s_tuple]))

        self.cache_interval[s] = ret
        return ret

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        assert b > 0
        ret = {}
        new_output = output
        score = 0
        for i in range(input_pos, len(s) + 1):
            if i > input_pos:
                if Alphabet.is_char_model:  # if character-level model
                    new_output += s[i - 1]
                else:
                    new_output += (s[i - 1],)
                end_pos = min(len(new_output) - 1, i - 1)
                score += np.sum(
                    partial_loss[end_pos] * (Alphabet.mapping[new_output[end_pos]] - Alphabet.mapping[s[end_pos]]))
            if Alphabet.is_char_model:  # if character-level model
                if self.regex.fullmatch(s[:i]):
                    ret[i] = [[new_output, score]]
            else:  # if word-level model
                if self.regex.fullmatch(Alphabet.escaped_char.join(s[:i])):
                    ret[i] = [[new_output, score]]

        return ret


class SWAP:
    def __init__(self, phi1, phi2):
        self.phi = [phi1, phi2]
        self.alphabet_acc = [Alphabet.get_acc_alphabet(phi) for phi in self.phi]

    def exact_space(self, s):
        if len(s) > 1 and self.phi[0](s[0]) and self.phi[1](s[1]):
            if Alphabet.is_char_model:  # if character-level model
                return {(2, s[1] + s[0])}
            else:  # if word-level model
                return {(2, (s[1], s[0]))}
        else:
            return set()

    def interval_space(self, s):
        ret = [(), ()]
        if len(s) > 0:
            for i in range(len(self.phi)):
                for single_s in s[i]:
                    if self.phi[i](single_s):
                        ret[1 - i] += (single_s,)

        return {} if len(ret[0]) == 0 or len(ret[1]) == 0 else {2: tuple(ret)}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        if input_pos + 1 < len(s) and self.phi[0](s[input_pos]) and self.phi[1](s[input_pos + 1]):
            if Alphabet.is_char_model:  # if character-level model
                new_output = output + s[input_pos + 1] + s[input_pos]
            else:
                new_output = output + (s[input_pos + 1], s[input_pos],)
            end_pos = min(len(new_output) - 1, input_pos + 1)
            score = 0
            for i in range(2):
                score += np.sum(partial_loss[end_pos - i] * (
                        Alphabet.mapping[new_output[end_pos - i]] - Alphabet.mapping[s[end_pos - i]]))
            return {input_pos + 2: [[new_output, score]]}
        else:
            return {}


class SUB:
    def __init__(self, phi, fun):
        self.phi = phi
        self.fun = fun
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)

    def exact_space(self, s):
        if len(s) > 0 and self.phi(s[0]):
            ret = set()
            tmp_ret = self.fun(s[0])
            for ss in tmp_ret:
                if Alphabet.is_char_model:  # if character-level model
                    ret.add((1, ss))
                else:  # if word-level model
                    ret.add((1, (ss,)))
            return ret
        else:
            return set()

    def interval_space(self, s):
        ret = ()
        if len(s) > 0:
            for single_s in s[0]:
                if self.phi(single_s):
                    tmp_ret = self.fun(single_s)
                    for ss in tmp_ret:
                        ret += (ss,)
        return {} if len(ret) == 0 else {1: (ret,)}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        if input_pos < len(s) and self.phi(s[input_pos]):
            ret = Beam(b)
            tmp_ret = self.fun(s[input_pos])
            for ss in tmp_ret:
                if Alphabet.is_char_model:  # if character-level model
                    new_output = output + ss
                else:
                    new_output = output + (ss,)
                end_pos = min(len(new_output) - 1, input_pos)
                score = np.sum(
                    partial_loss[end_pos] * (Alphabet.mapping[new_output[end_pos]] - Alphabet.mapping[s[end_pos]]))
                ret.add(new_output, score)
            return {input_pos + 1: ret.check_balance()}
        else:
            return {}


class tUnion:
    def __init__(self, *arg):
        self.t = arg
        for x in self.t:
            assert isinstance(x, REGEX) or isinstance(x, DEL) or isinstance(x, INS) or isinstance(x, SWAP) \
                   or isinstance(x, SUB) or isinstance(x, tUnion)

    def exact_space(self, s):
        ret = set()
        for t in self.t:
            ret.update(t.exact_space(s))
        return ret

    def interval_space(self, s):
        ret = {}
        for t in self.t:
            tmp_ret = t.interval_space(s)
            for pos in tmp_ret:
                if pos not in ret:
                    ret[pos] = tmp_ret[pos]
                else:
                    ret[pos] = tuple_set_union(ret[pos], tmp_ret[pos])

        return ret

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        '''
        The beam search for tUnion
        :param s: the input string s
        :param output: the current output
        :param input_pos: the current length of input already taken
        :param b: the budget b
        :param partial_with_respect_to_loss: the partial derivative of loss with respect to s
        :return: a dict, the keys are the length of input consumed, the values are [output, score],
            output is the produced output of tUnion included previous output, the score is the beam search's score.
        '''
        ret = {}
        for t in self.t:
            tmp_ret = t.beam_search_adversarial(s, output, input_pos, b, partial_loss)
            for pos in tmp_ret:
                if pos not in ret:
                    ret[pos] = Beam(b)
                ret[pos].extend(tmp_ret[pos])

        for pos in ret:
            ret[pos] = ret[pos].check_balance()
        return ret


class Transformation:
    def __init__(self, *arg):
        self.seq = arg
        self.cache_exact = {}
        self.cache_interval = {}
        for x in self.seq:
            assert isinstance(x, REGEX) or isinstance(x, DEL) or isinstance(x, INS) or isinstance(x, SWAP) \
                   or isinstance(x, SUB) or isinstance(x, tUnion)

    def exact_space(self, s):
        if s in self.cache_exact:
            return self.cache_exact[s]
        ret = set()
        if Alphabet.is_char_model:  # if character-level model
            ret.add((0, ""))
        else:  # if word-level model
            ret.add((0, ()))
        for t in self.seq:
            new_ret = set()
            for pos, output in ret:
                tmp_res = t.exact_space(s[pos:])
                for new_pos, new_output in tmp_res:
                    new_ret.add((pos + new_pos, output + new_output))
            ret = new_ret

        ret_without_pos = set()
        for pos, output in ret:
            if pos == len(s):
                ret_without_pos.add(output)
        self.cache_exact[s] = ret_without_pos
        return ret_without_pos

    def interval_space(self, s):
        if s in self.cache_interval:
            return self.cache_interval[s]
        ret = {0: ()}
        for (pos_seq, t) in enumerate(self.seq):
            new_ret = {}
            for pos in ret:
                output = ret[pos]
                tmp_res = t.interval_space(s[pos:])
                for new_pos in tmp_res:
                    new_output = tmp_res[new_pos]
                    final_pos = new_pos + pos
                    # check whether the afterwards chars/words are accepted by afterwords transformations
                    if final_pos in new_ret:
                        new_ret[final_pos] = tuple_set_union(new_ret[final_pos], output + new_output)
                    else:
                        new_ret[final_pos] = output + new_output
            ret = new_ret

        final_ret = None

        if len(s) in ret:
            final_ret = ret[len(s)]
        for (i, single_s) in enumerate(s):
            if "" in single_s:
                final_ret = tuple_set_union(final_ret, ret[i])

        self.cache_interval[s] = final_ret
        return final_ret

    def beam_search_adversarial(self, s, y_true, b):
        '''
        Beam search for the transformation. Not sure should the beam search be used here.
        The HotFlip paper uses brute-force enumerate. Or it uses the beam search with b>0.
            (since only there is only one beam in terms of position).
        If we used beam search here: it can boost the speed, but downgrade the adversarial examples
        This implementation used the beam search with budget the same as the outside beam search budget, b
        :param s: the input string s
        :param b: the budget for the beam search
        :return: the a list of adversarial examples within budget b
        '''
        inner_budget = b  # can be adjusted to meet the user's demand, set np.inf to do brute_force enumeration
        partial_loss = Alphabet.partial_to_loss(Alphabet.to_embedding(s), y_true)
        ret = {}
        if Alphabet.is_char_model:  # if character-level model
            ret[0] = [["", 0]]
        else:  # if word-level model
            ret[0] = [[(), 0]]
        for t in self.seq:
            new_ret = {}
            for pos in ret:
                for output, score in ret[pos]:
                    tmp_res = t.beam_search_adversarial(s, output, pos, inner_budget, partial_loss)
                    for new_pos in tmp_res:
                        if new_pos not in new_ret:
                            new_ret[new_pos] = Beam(inner_budget)
                        for new_output, new_score in tmp_res[
                            new_pos]:  # the new output for all previous unit transformation
                            new_ret[new_pos].add(new_output, score + new_score)

            for pos in new_ret:
                new_ret[pos] = new_ret[pos].check_balance()
            ret = new_ret

        if len(s) in ret:
            true_ret = Beam(b)
            for data, score in ret[len(s)]:
                for i in range(len(data), len(s)):
                    score += np.sum(partial_loss[i] * (Alphabet.mapping[Alphabet.padding] - Alphabet.mapping[s[i]]))
                true_ret.add(data, score)
            return true_ret.check_balance()
        else:
            return []


class Union:
    def __init__(self, *args):
        self.p = args
        assert len(args) > 0
        self.cache_exact = {}
        self.cache_interval = {}
        for p in self.p:
            assert isinstance(p, Transformation) or isinstance(p, Union) or isinstance(p, Composition)

    def exact_space(self, s):
        if s in self.cache_exact:
            return self.cache_exact[s]
        ret = set()
        for p in self.p:
            ret.update(p.exact_space(s))
        self.cache_exact[s] = ret
        return ret

    def interval_space(self, s):
        if s in self.cache_interval:
            return self.cache_interval[s]
        ret = None
        for p in self.p:
            ret = tuple_set_union(ret, p.interval_space(s))
        self.cache_interval[s] = ret
        return ret

    def beam_search_adversarial(self, s, y_true, b):
        ret = Beam(b)
        for p in self.p:
            ret.extend(p.beam_search_adversarial(s, y_true, b))
        return ret.check_balance()


class Composition:
    def __init__(self, *args):
        self.p = args  # should be reversed
        assert len(args) > 0
        self.cache_exact = {}
        self.cache_interval = {}
        for p in self.p:
            assert isinstance(p, Transformation) or isinstance(p, Union) or isinstance(p, Composition)

    def exact_space(self, s):
        if s in self.cache_exact:
            return self.cache_exact[s]
        ret = {s}
        for p in reversed(self.p):
            new_ret = set()
            for s in ret:
                new_ret.update(p.exact_space(s))
            ret = new_ret
        self.cache_exact[s] = ret
        return ret

    def interval_space(self, s):
        if s in self.cache_interval:
            return self.cache_interval[s]
        ret = s
        for p in reversed(self.p):
            ret = p.interval_space(ret)
        self.cache_interval[s] = ret
        return ret

    def beam_search_adversarial(self, s, y_true, b):
        '''
        Beam search for adversarial examples within budget b.
        :param s: the input s
        :param b: the budget b
        :return: the
        '''
        ret = [[s, 0]]
        for p in reversed(self.p):
            new_ret = Beam(b)
            for s, score in ret:
                new_ret.extend([[x, y + score] for (x, y) in p.beam_search_adversarial(s, y_true, b)])
            ret = new_ret.check_balance()

        return ret
