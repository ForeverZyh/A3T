import re
import itertools
import numpy as np
import multiprocessing.connection
from nltk import pos_tag

from utils import tuple_set_union, Beam, UnorderedBeam, inf
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
            if end_pos == len(new_output) - 1 and end_pos < Alphabet.max_len:
                score = np.sum(
                    partial_loss[end_pos] * (
                            Alphabet.embedding[Alphabet.mapping[new_output[end_pos]]] - Alphabet.embedding[
                        Alphabet.mapping[s[end_pos]]]))
            else:
                score = 0
            ret.add(new_output, score)

        return {input_pos: ret.check_balance()}

    def random_sample(self, s, output, input_pos, b):
        assert b > 0
        ret = UnorderedBeam(b)
        for c in self.alphabet_acc:
            if Alphabet.is_char_model:  # if character-level model
                new_output = output + c
            else:  # if word-level model
                new_output = output + (c,)
            ret.add(new_output)

        return {input_pos: ret.check_balance()}


class DUP:
    def __init__(self, phi, fun):
        self.phi = phi
        self.fun = fun
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)
        self.alphabet_acc_set = set(self.alphabet_acc)

    def exact_space(self, s):
        ret = set()
        if len(s) > 0 and s[0] in self.alphabet_acc_set:
            dup_set = self.fun(s[0])
            for c in dup_set:
                if Alphabet.is_char_model:  # if character-level model
                    ret.add((1, s[0] + c))
                else:  # if word-level model
                    ret.add((1, (s[0], c)))
        return ret

    def interval_space(self, s):
        if len(s) > 0 and not set(self.alphabet_acc).isdisjoint(set(s[0])):
            return {1: ((s[0],), tuple(self.fun(s[0])))}
        else:
            return {}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        assert b > 0
        ret = Beam(b)
        if input_pos < len(s) and s[input_pos] in self.alphabet_acc_set:
            dup_set = self.fun(s[input_pos])
            for c in dup_set:
                pre_pos = min(len(output), input_pos)
                if Alphabet.is_char_model:  # if character-level model
                    new_output = output + s[input_pos] + c
                else:  # if word-level model
                    new_output = output + (s[input_pos], c,)
                end_pos = min(len(new_output), input_pos + 1)
                score = 0
                for pos in range(pre_pos, end_pos):
                    if pos >= Alphabet.max_len: continue
                    score += np.sum(
                        partial_loss[pos] * (
                                Alphabet.embedding[Alphabet.mapping[new_output[pos]]] - Alphabet.embedding[
                            Alphabet.mapping[s[pos]]]))

                ret.add(new_output, score)

        return {input_pos + 1: ret.check_balance()}

    def random_sample(self, s, output, input_pos, b):
        assert b > 0
        ret = UnorderedBeam(b)
        if input_pos < len(s) and s[input_pos] in self.alphabet_acc_set:
            dup_set = self.fun(s[input_pos])
            for c in dup_set:
                if Alphabet.is_char_model:  # if character-level model
                    new_output = output + s[input_pos] + c
                else:  # if word-level model
                    new_output = output + (s[input_pos], c)
                ret.add(new_output)

        return {input_pos + 1: ret.check_balance()}


class DEL:
    def __init__(self, phi):
        self.phi = phi
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)
        self.alphabet_acc_set = set(self.alphabet_acc)

    def exact_space(self, s):
        if len(s) > 0 and s[0] in self.alphabet_acc_set:
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
        if input_pos < len(s) and s[input_pos] in self.alphabet_acc_set:
            end_pos = min(len(output) - 1, input_pos)
            if end_pos == input_pos:
                score = np.sum(partial_loss[end_pos] * (
                        Alphabet.embedding[Alphabet.mapping[output[end_pos]]] - Alphabet.embedding[
                    Alphabet.mapping[s[end_pos]]]))
            else:
                score = 0
            return {input_pos + 1: [[output, score]]}
        else:
            return {}

    def random_sample(self, s, output, input_pos, b):
        assert b > 0
        if input_pos < len(s) and s[input_pos] in self.alphabet_acc_set:
            return {input_pos + 1: [output]}
        else:
            return {}


class REGEX:
    def __init__(self, regex):
        self.is_any = regex == r'.*'
        self.regex = re.compile(regex)
        self.cache_exact = {}
        self.cache_interval = {}

    def exact_space(self, s, is_end=False):
        if (s, is_end) in self.cache_exact:
            return self.cache_exact[(s, is_end)]
        ret = set()
        for i in range(len(s) + 1):
            if not is_end or (is_end and i == len(s)):
                if Alphabet.is_char_model:  # if character-level model
                    if self.is_any or self.regex.fullmatch(s[:i]):
                        ret.add((i, s[:i]))
                else:  # if word-level model
                    if self.is_any or self.regex.fullmatch(Alphabet.escaped_char.join(s[:i])):
                        ret.add((i, s[:i]))
        self.cache_exact[(s, is_end)] = ret
        return ret

    def interval_space(self, s, is_end=False):
        # TODO figure out a more efficient way to do this.
        #  One optimization maybe use DFA so that some prefix can be eliminated at early stage, and the exponential
        #  enumeration will be merged by their quotients (mapping to the same state on the DFA).
        #  Then the exponential enumeration becomes polynomial of #state of the DFA.
        #  Currently we only optimized the ".*".
        if (s, is_end) in self.cache_interval:
            return self.cache_interval[(s, is_end)]
        ret = {}
        for i in range(len(s) + 1):
            if not is_end or (is_end and (i == len(s) or "" in s[i])):
                if not self.is_any:
                    for s_tuple in itertools.product(*(s[:i])):
                        if (Alphabet.is_char_model and self.regex.fullmatch("".join(s_tuple))) or (
                                not Alphabet.is_char_model and self.regex.fullmatch(
                            Alphabet.escaped_char.join(s_tuple))):
                            if i not in ret:
                                ret[i] = tuple([(t,) for t in s_tuple])
                            else:
                                ret[i] = tuple_set_union(ret[i], tuple([(t,) for t in s_tuple]))
                else:
                    ret[i] = s[:i]

        self.cache_interval[(s, is_end)] = ret
        return ret

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss, is_end=False):
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
                if input_pos != len(output):
                    end_pos = min(len(new_output) - 1, i - 1)
                    if end_pos < Alphabet.max_len:
                        score += np.sum(partial_loss[end_pos] * (
                                Alphabet.embedding[Alphabet.mapping[new_output[end_pos]]] - Alphabet.embedding[
                            Alphabet.mapping[s[end_pos]]]))
            if not is_end or (is_end and i == len(s)):
                if Alphabet.is_char_model:  # if character-level model
                    if self.is_any or self.regex.fullmatch(s[:i]):
                        ret[i] = [[new_output, score]]
                else:  # if word-level model
                    if self.is_any or self.regex.fullmatch(Alphabet.escaped_char.join(s[:i])):
                        ret[i] = [[new_output, score]]

        return ret

    def random_sample(self, s, output, input_pos, b, is_end=False):
        assert b > 0
        ret = {}
        new_output = output
        for i in range(input_pos, len(s) + 1):
            if i > input_pos:
                if Alphabet.is_char_model:  # if character-level model
                    new_output += s[i - 1]
                else:
                    new_output += (s[i - 1],)
            if not is_end or (is_end and i == len(s)):
                if Alphabet.is_char_model:  # if character-level model
                    if self.is_any or self.regex.fullmatch(s[:i]):
                        ret[i] = [new_output]
                else:  # if word-level model
                    if self.is_any or self.regex.fullmatch(Alphabet.escaped_char.join(s[:i])):
                        ret[i] = [new_output]

        return ret


class SWAP:
    def __init__(self, phi1, phi2):
        self.phi = [phi1, phi2]
        self.alphabet_acc = [Alphabet.get_acc_alphabet(phi) for phi in self.phi]
        self.alphabet_acc_set = [set(t) for t in self.alphabet_acc]

    def exact_space(self, s):
        if len(s) > 1 and s[0] in self.alphabet_acc_set[0] and s[1] in self.alphabet_acc_set[1]:
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
                ret[1 - i] = tuple(set(s[i]).intersection(self.alphabet_acc_set[i]))

        return {} if len(ret[0]) == 0 or len(ret[1]) == 0 else {2: tuple(ret)}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        assert b > 0
        if input_pos + 1 < len(s) and s[input_pos] in self.alphabet_acc_set[0] and s[input_pos + 1] in \
                self.alphabet_acc_set[1] and s[input_pos] != s[input_pos + 1]:
            if Alphabet.is_char_model:  # if character-level model
                new_output = output + s[input_pos + 1] + s[input_pos]
            else:
                new_output = output + (s[input_pos + 1], s[input_pos],)
            end_pos = min(len(new_output) - 1, input_pos + 1)
            score = 0
            for i in range(2):
                score += np.sum(partial_loss[end_pos - i] * (
                        Alphabet.embedding[Alphabet.mapping[new_output[end_pos - i]]] - Alphabet.embedding[
                    Alphabet.mapping[s[end_pos - i]]]))
            return {input_pos + 2: [[new_output, score]]}
        else:
            return {}

    def random_sample(self, s, output, input_pos, b):
        assert b > 0
        if input_pos + 1 < len(s) and s[input_pos] in self.alphabet_acc_set[0] and s[input_pos + 1] in \
                self.alphabet_acc_set[1]:
            if Alphabet.is_char_model:  # if character-level model
                new_output = output + s[input_pos + 1] + s[input_pos]
            else:
                new_output = output + (s[input_pos + 1], s[input_pos],)
            return {input_pos + 2: [new_output]}
        else:
            return {}


class SUB:
    def __init__(self, phi, fun, add_fun=None):
        self.phi = phi
        self.fun = fun
        self.add_fun = add_fun
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)
        self.alphabet_acc_set = set(self.alphabet_acc)

    def exact_space(self, s):
        if len(s) > 0 and s[0] in self.alphabet_acc_set:
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
        ret = set()
        if len(s) > 0:
            for single_s in s[0]:
                if single_s in self.alphabet_acc_set:
                    tmp_ret = self.fun(single_s)
                    ret.update(tmp_ret)

        return {} if len(ret) == 0 else {1: (tuple(ret),)}

    def beam_search_adversarial(self, s, output, input_pos, b, partial_loss):
        assert b > 0
        if input_pos < len(s) and s[input_pos] in self.alphabet_acc_set:
            # input_pos_tag = pos_tag(s)[input_pos][1]
            ret = Beam(1)
            tmp_ret = self.fun(s[input_pos])
            add_tmp_ret = None
            if self.add_fun is not None:
                add_tmp_ret = self.add_fun(s[input_pos])
                input_pos_tag = pos_tag(s)[input_pos][1]
            else:
                input_pos_tag = None
            for (i, ss) in enumerate(tmp_ret):
                if add_tmp_ret is None or add_tmp_ret[i] == input_pos_tag:
                    if Alphabet.is_char_model:  # if character-level model
                        new_output = output + ss
                    else:
                        new_output = output + (ss,)
                    end_pos = min(len(new_output) - 1, input_pos)
                    score = np.sum(
                        partial_loss[end_pos] * (
                                Alphabet.embedding[Alphabet.mapping[new_output[end_pos]]] - Alphabet.embedding[
                            Alphabet.mapping[s[end_pos]]]))
                    ret.add(new_output, score)
            return {input_pos + 1: ret.check_balance()}
        else:
            return {}

    def random_sample(self, s, output, input_pos, b):
        assert b > 0
        if input_pos < len(s) and s[input_pos] in self.alphabet_acc_set:
            ret = UnorderedBeam(1)
            tmp_ret = self.fun(s[input_pos])
            for ss in tmp_ret:
                if Alphabet.is_char_model:  # if character-level model
                    new_output = output + ss
                else:
                    new_output = output + (ss,)
                ret.add(new_output)
            return {input_pos + 1: ret.check_balance()}
        else:
            return {}


class tUnion:
    def __init__(self, *arg):
        self.t = arg
        # whether the tUnion has certain type of local transformations
        self.has_swap = False
        self.has_ins = False
        self.has_del = False
        self.has_sub = False
        for x in self.t:
            if isinstance(x, SWAP):
                self.has_swap = True
            elif isinstance(x, INS):
                self.has_ins = True
            elif isinstance(x, DEL):
                self.has_del = True
            elif isinstance(x, SUB):
                self.has_sub = True
            elif isinstance(x, tUnion):
                self.has_swap |= x.has_swap
                self.has_ins |= x.has_ins
                self.has_del |= x.has_del
                self.has_sub |= x.has_sub

            assert isinstance(x, REGEX) or isinstance(x, DEL) or isinstance(x, INS) or isinstance(x, SWAP) \
                   or isinstance(x, SUB) or isinstance(x, tUnion) or isinstance(x, DUP)

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

    def random_sample(self, s, output, input_pos, b):
        ret = {}
        for t in self.t:
            tmp_ret = t.random_sample(s, output, input_pos, b)
            for pos in tmp_ret:
                if pos not in ret:
                    ret[pos] = UnorderedBeam(b)
                ret[pos].extend(tmp_ret[pos])

        for pos in ret:
            ret[pos] = ret[pos].check_balance()
        return ret


class Transformation:
    def __init__(self, *arg, **kwargs):
        self.seq = arg
        self.cache_exact = {}
        self.cache_interval = {}
        self.delta = None
        self.max_delta = None
        self.max_increment = None
        if "inner_budget" in kwargs:
            self.inner_budget = kwargs["inner_budget"]
        else:
            self.inner_budget = None
        if "truncate" in kwargs and kwargs[
            "truncate"] is not None:  # TODO now it only affects the beam_search_adversarial
            self.truncate = kwargs["truncate"]
        else:
            self.truncate = inf
        for x in self.seq:
            assert isinstance(x, REGEX) or isinstance(x, DEL) or isinstance(x, INS) or isinstance(x, SWAP) \
                   or isinstance(x, SUB) or isinstance(x, tUnion) or isinstance(x, DUP)

    def exact_space(self, s):
        if s in self.cache_exact:
            return self.cache_exact[s]
        ret = set()
        if Alphabet.is_char_model:  # if character-level model
            ret.add((0, ""))
        else:  # if word-level model
            ret.add((0, ()))
        for (i_seq, t) in enumerate(self.seq):
            new_ret = set()
            for pos, output in ret:
                if isinstance(t, REGEX):
                    tmp_res = t.exact_space(s[pos:], i_seq + 1 == len(self.seq))
                else:
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
                if isinstance(t, REGEX):
                    tmp_res = t.interval_space(s[pos:], pos_seq + 1 == len(self.seq))
                else:
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
        inner_budget = b if self.inner_budget is None else self.inner_budget  # can be adjusted to meet the user's demand, set np.inf to do brute_force enumeration
        if isinstance(y_true, multiprocessing.connection.Connection):
            y_true.send(Alphabet.to_ids(s))
            partial_loss = y_true.recv()
        else:
            partial_loss = Alphabet.partial_to_loss(Alphabet.to_ids(s), y_true)

        ret = {}
        if Alphabet.is_char_model:  # if character-level model
            ret[0] = [["", 0]]
        else:  # if word-level model
            ret[0] = [[(), 0]]
        for (i_seq, t) in enumerate(self.seq):
            new_ret = {}
            for pos in ret:
                for output, score in ret[pos]:
                    if isinstance(t, REGEX):
                        tmp_res = t.beam_search_adversarial(s, output, pos, inner_budget, partial_loss,
                                                            i_seq + 1 == len(self.seq))
                    else:
                        tmp_res = t.beam_search_adversarial(s, output, pos, inner_budget, partial_loss)
                    for new_pos in tmp_res:
                        if new_pos > self.truncate and (not isinstance(t, REGEX) or not t.is_any):
                            # if the unit-transformation exceeds the truncated length and
                            # it is not a REGEX or it is a not any REGEX
                            continue
                        if new_pos not in new_ret:
                            new_ret[new_pos] = Beam(inner_budget)
                        for new_output, new_score in tmp_res[
                            new_pos]:  # the new output for all previous unit transformation
                            if not new_ret[new_pos].add(new_output, score + new_score):
                                break

            for pos in new_ret:
                new_ret[pos] = new_ret[pos].check_balance()
            ret = new_ret

        true_ret = Beam(b)
        true_ret.add(s, 0)
        if len(s) in ret:
            for data, score in ret[len(s)]:
                pre_pos = min(len(data), len(s))
                end_pos = min(max(len(data), len(s)), Alphabet.max_len)
                for i in range(pre_pos, end_pos):
                    score += np.sum(partial_loss[i] * (
                            Alphabet.embedding[Alphabet.mapping[data[i] if i < len(data) else Alphabet.padding]] -
                            Alphabet.embedding[Alphabet.mapping[s[i] if i < len(s) else Alphabet.padding]]))
                true_ret.add(data, score)

        return true_ret.check_balance()

    def random_sample(self, s, b):
        inner_budget = b if self.inner_budget is None else self.inner_budget  # can be adjusted to meet the user's demand, set np.inf to do brute_force enumeration
        ret = {}
        if Alphabet.is_char_model:  # if character-level model
            ret[0] = [""]
        else:  # if word-level model
            ret[0] = [()]
        for (i_seq, t) in enumerate(self.seq):
            new_ret = {}
            for pos in ret:
                for output in ret[pos]:
                    if isinstance(t, REGEX):
                        tmp_res = t.random_sample(s, output, pos, inner_budget, i_seq + 1 == len(self.seq))
                    else:
                        tmp_res = t.random_sample(s, output, pos, inner_budget)
                    for new_pos in tmp_res:
                        if new_pos not in new_ret:
                            new_ret[new_pos] = UnorderedBeam(inner_budget)
                        for new_output in tmp_res[new_pos]:  # the new output for all previous unit transformation
                            new_ret[new_pos].add(new_output)

            for pos in new_ret:
                new_ret[pos] = new_ret[pos].check_balance()
            ret = new_ret

        true_ret = UnorderedBeam(b)
        true_ret.add(s)
        if len(s) in ret:
            for data in ret[len(s)]:
                true_ret.add(data)

        return true_ret.check_balance()

    def get_delta(self):
        if self.delta is None:
            self.delta = 0
            # we only compute the delta regarding to substitution and swap.
            for t in self.seq:
                if isinstance(t, SUB) or (isinstance(t, tUnion) and t.has_sub):
                    self.delta += 1
                elif isinstance(t, SWAP) or (isinstance(t, tUnion) and t.has_swap):
                    self.delta += 2

        return self.delta

    def get_max_increment(self):
        if self.max_increment is None:
            self.max_increment = [False, 0]
            for t in self.seq:
                if isinstance(t, INS) or (isinstance(t, tUnion) and t.has_ins):
                    self.max_increment[0] = True
                    self.max_increment[1] += 1
                elif isinstance(t, DEL) or (isinstance(t, tUnion) and t.has_del):
                    # Notice this elif instead if: we first consider insertion in the tUnion since we want to compute
                    # the max increment
                    self.max_increment[0] = True
                    self.max_increment[1] -= 1

        return self.max_increment

    def get_max_delta(self):
        return get_max_delta(self)


class TransformationDel(Transformation):
    def __init__(self, *arg, **kwargs):
        if "inner_budget" in kwargs:
            self.inner_budget = kwargs["inner_budget"]
        else:
            self.inner_budget = None

        if "truncate" in kwargs and kwargs["truncate"] is not None:
            self.truncate = kwargs["truncate"]
        else:
            self.truncate = inf

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
        inner_budget = b if self.inner_budget is None else self.inner_budget  # can be adjusted to meet the user's demand, set np.inf to do brute_force enumeration
        if isinstance(y_true, multiprocessing.connection.Connection):
            y_true.send(Alphabet.to_ids(s))
            partial_loss = y_true.recv()
        else:
            partial_loss = Alphabet.partial_to_loss(Alphabet.to_ids(s), y_true)

        sum_suffix = []
        for i in range(len(s) - 1):
            sum_suffix.append(np.sum(partial_loss[i] * (
                    Alphabet.embedding[Alphabet.mapping[s[i + 1]]] - Alphabet.embedding[Alphabet.mapping[s[i]]])))
        sum_suffix.append(np.sum(partial_loss[len(s) - 1] * (
                Alphabet.embedding[Alphabet.mapping[Alphabet.padding]] - Alphabet.embedding[
            Alphabet.mapping[s[len(s) - 1]]])))
        for i in range(len(sum_suffix) - 2, -1, -1):
            sum_suffix[i] += sum_suffix[i + 1]

        true_ret = Beam(b)
        true_ret.add(s, 0)
        delete_pos = [i for i in range(min(len(s), self.truncate))]
        delete_pos.sort(key=lambda x: -sum_suffix[x])
        for i in range(min(b, len(delete_pos))):
            pos = delete_pos[i]
            true_ret.add(s[:pos] + s[pos + 1:], sum_suffix[pos])

        return true_ret.check_balance()


class TransformationIns(Transformation):
    def __init__(self, *arg, **kwargs):
        if "inner_budget" in kwargs:
            self.inner_budget = kwargs["inner_budget"]
        else:
            self.inner_budget = None

        if "truncate" in kwargs and kwargs["truncate"] is not None:
            self.truncate = kwargs["truncate"]
        else:
            self.truncate = inf

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
        inner_budget = b if self.inner_budget is None else self.inner_budget  # can be adjusted to meet the user's demand, set np.inf to do brute_force enumeration
        if isinstance(y_true, multiprocessing.connection.Connection):
            y_true.send(Alphabet.to_ids(s))
            partial_loss = y_true.recv()
        else:
            partial_loss = Alphabet.partial_to_loss(Alphabet.to_ids(s), y_true)

        sum_suffix = [0]
        for i in range(1, len(s)):
            sum_suffix.append(np.sum(partial_loss[i] * (
                    Alphabet.embedding[Alphabet.mapping[s[i - 1]]] - Alphabet.embedding[Alphabet.mapping[s[i]]])))
        if len(s) < Alphabet.max_len:
            sum_suffix.append(np.sum(partial_loss[len(s)] * (
                    Alphabet.embedding[Alphabet.mapping[s[len(s) - 1]]] - Alphabet.embedding[
                Alphabet.mapping[Alphabet.padding]])))
        sum_suffix.append(0)
        for i in range(len(sum_suffix) - 2, -1, -1):
            sum_suffix[i] += sum_suffix[i + 1]
        cost = [-10000 for i in range(len(s))]
        if len(s) < Alphabet.max_len:
            for i in range(len(s)):
                if s[i] in Alphabet.adjacent_keys:
                    t = list(Alphabet.adjacent_keys[s[i]])[0]
                    if i + 1 < len(s):
                        cost[i] = sum_suffix[i + 2] + np.sum(partial_loss[i + 1] * (
                                Alphabet.embedding[Alphabet.mapping[t]] - Alphabet.embedding[
                            Alphabet.mapping[s[i + 1]]]))
                    else:
                        cost[i] = sum_suffix[i + 2] + np.sum(partial_loss[i + 1] * (
                                Alphabet.embedding[Alphabet.mapping[t]] - Alphabet.embedding[
                            Alphabet.mapping[Alphabet.padding]]))
        else:
            for i in range(len(s) - 1):
                if s[i] in Alphabet.adjacent_keys:
                    t = list(Alphabet.adjacent_keys[s[i]])[0]
                    cost[i] = sum_suffix[i + 2] + np.sum(partial_loss[i + 1] * (
                            Alphabet.embedding[Alphabet.mapping[t]] - Alphabet.embedding[Alphabet.mapping[s[i + 1]]]))

        true_ret = Beam(b)
        true_ret.add(s, 0)
        ins_pos = [i for i in range(min(len(s), self.truncate)) if s[i] in Alphabet.adjacent_keys]
        ins_pos.sort(key=lambda x: -cost[x])
        for i in range(min(b, len(ins_pos))):
            pos = ins_pos[i]
            # if cost[pos] < 0:
            #    break
            t = list(Alphabet.adjacent_keys[s[pos]])[0]
            true_ret.add((s[:pos + 1] + t + s[pos + 1:])[:Alphabet.max_len], cost[pos])

        return true_ret.check_balance()


class Union:
    def __init__(self, *args):
        self.p = args
        assert len(args) > 0
        self.cache_exact = {}
        self.cache_interval = {}
        self.delta = None
        self.max_delta = None
        self.max_increment = None
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

    def random_sample(self, s, b):
        ret = UnorderedBeam(b)
        for p in self.p:
            ret.extend(p.random_sample(s, b))
        return ret.check_balance()

    def get_delta(self):
        if self.delta is None:
            self.delta = 0
            # the max delta of union is the max of them,
            # e.g., S1-> delta=2 ->S2; S1-> delta=1 ->S2, then S1-> delta=max(2,1)=2 -> S2.
            for p in self.p:
                self.delta = max(p.get_delta(), self.delta)

        return self.delta

    def get_max_increment(self):
        if self.max_increment is None:
            self.max_increment = [False, 0]
            for p in self.p:
                has_ins_delta, max_increment = p.get_max_increment()
                # we get the maximum of increment in Union
                if has_ins_delta:
                    if self.max_increment[0]:
                        self.max_increment[1] = max(max_increment, self.max_increment[1])
                    else:
                        self.max_increment = [has_ins_delta, max_increment]

        return self.max_increment

    def get_max_delta(self):
        return get_max_delta(self)


class Composition:
    def __init__(self, *args):
        self.p = args  # should be reversed
        assert len(args) > 0
        self.cache_exact = {}
        self.cache_interval = {}
        self.delta = None
        self.max_delta = None
        self.max_increment = None
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
            if ret is None:
                break
            ret = p.interval_space(ret)
        self.cache_interval[s] = ret
        return ret

    def beam_search_adversarial(self, s, y_true, b):
        '''
        Beam search for adversarial examples within budget b.
        :param s: the input s
        :param b: the budget b
        :return: the a list of adversarial examples within budget b
        '''
        ret = [[s, 0]]
        for p in reversed(self.p):
            new_ret = Beam(b)
            for s, score in ret:
                new_ret.extend([[x, y + score] for (x, y) in p.beam_search_adversarial(s, y_true, b)])
            ret = new_ret.check_balance()

        return ret

    def random_sample(self, s, b):
        ret = [s]
        for p in reversed(self.p):
            new_ret = UnorderedBeam(b)
            for s in ret:
                new_ret.extend(p.random_sample(s, b))
            ret = new_ret.check_balance()

        return ret

    def get_delta(self):
        if self.delta is None:
            self.delta = 0
            # the delta of composition is added together,
            # e.g., S1-> delta=2 ->S2; S2-> delta=1 ->S3, then S1-> delta=2+1=3 -> S3.
            for p in reversed(self.p):
                self.delta += p.get_delta()

        return self.delta

    def get_max_increment(self):
        '''
        Get the maximum increment of the output after the transformation
        :return: [has_ins_delta: Bool, max_increment: int]
        has_ins_delta indicates whether the transformation contains insertions or deletions
        max_increment is the maximum increment of the output after the transformation
        '''
        if self.max_increment is None:
            self.max_increment = [False, 0]
            for p in reversed(self.p):
                has_ins_delta, max_increment = p.get_max_increment()
                if has_ins_delta:
                    self.max_increment[0] = True
                    # the increment is adding together in Composition
                    self.max_increment[1] += max_increment

        return self.max_increment

    def get_max_delta(self):
        return get_max_delta(self)


def get_max_delta(self):
    '''
    Get the maximum delta of a transformation
    :return: [is_max_len, delta],
        is_max_len: is an 0/1 integer indicating whether the max_delta is related to the length of the input without
        padding (len_wo_padding).
        delta: is the size of perturbation regarding to either with/without len_wo_padding.
        The max_delta can be calculated as is_max_len*len_wo_padding + delta.
        if the transformation only has one deletion, then (is_max_len, delta)=(1, 0);
        if the transformation only has one insertion, then (is_max_len, delta)=(1, 1);
    '''
    if self.max_delta is None:
        # We first compute the maximum increment of length of the output without padding.
        # The key is to count #ins-#del.
        has_ins_del, max_increment = self.get_max_increment()
        if has_ins_del:
            self.max_delta = [1, max(max_increment, 0)]
        else:
            self.max_delta = [0, self.get_delta()]

    return self.max_delta
