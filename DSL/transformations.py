import re
import itertools


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


class Alphabet:
    alphabet = []
    escaped_char = ""
    is_char_model = None
    embedding = None

    @staticmethod
    def set_alphabet(a: dict, embedding):
        Alphabet.alphabet = [None] * len(a)
        for s in a:
            Alphabet.alphabet[a[s]] = s
        Alphabet.embedding = embedding
        assert len(embedding) == len(a)

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

    # def check_sat(self, single_s):
    #     return not set(self.alphabet_acc).isdisjoint(set(single_s))


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

    # def check_sat(self, s):
    #     # TODO the same as the above one
    #     if s in self.cache:
    #
    #     for s_list in itertools.product(*s):
    #         if Alphabet.is_char_model:
    #             if self.regex.fullmatch("".join(s_list)):
    #                 return True
    #         else:
    #             if self.regex.fullmatch(Alphabet.escaped_char.join(s_list)):
    #                 return True
    #     return False


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

    # def check_sat(self, *single_s):
    #     for i in range(len(self.phi)):
    #         if set(single_s[i]).isdisjoint(set(self.alphabet_acc[i])):
    #             return False
    #     return True


class SUB:
    def __init__(self, phi, fun):
        self.phi = phi
        self.fun = fun
        self.alphabet_acc = Alphabet.get_acc_alphabet(self.phi)

    def exact_space(self, s):
        if len(s) > 0 and self.phi(s[0]):
            if Alphabet.is_char_model:  # if character-level model
                return {(1, self.fun(s[0]))}
            else:  # if word-level model
                return {(1, (self.fun(s[0]),))}
        else:
            return set()

    def interval_space(self, s):
        ret = ()
        if len(s) > 0:
            for single_s in s[0]:
                if self.phi(single_s):
                    ret += (self.fun(single_s),)
        return {} if len(ret) == 0 else {1: (ret,)}

    # def check_sat(self, single_s):
    #     return not set(single_s).isdisjoint(set(self.alphabet_acc))


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

    # def check_sat(self, s):
    #     pass


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

    # def check_sat(self, pos_seq, s):
    #     if (pos_seq, s) in self.cache_check_sat:
    #         return self.cache_check_sat[(pos_seq, s)]
    #     if pos_seq == len(self.seq):
    #         epsilon_exists = True
    #         for ss in s:
    #             if "" not in ss:
    #                 epsilon_exists = False
    #                 break
    #         self.cache_check_sat[(pos_seq, s)] = epsilon_exists
    #         return epsilon_exists
    #     else:
    #         ret = False
    #         if isinstance(self.seq[pos_seq], DEL):
    #             ret = len(s) > 0 and self.seq[pos_seq].check_sat(s[0]) and self.check_sat(pos_seq + 1, s[1:])
    #         elif isinstance(self.seq[pos_seq], INS):
    #             ret = len(s) == 0 or self.check_sat(pos_seq + 1, s[1:])
    #         elif isinstance(self.seq[pos_seq], SUB):
    #             ret = len(s) > 0 and self.seq[pos_seq].check_sat(s[0]) and self.check_sat(pos_seq + 1, s[1:])
    #         elif isinstance(self.seq[pos_seq], SWAP):
    #             ret = len(s) > 1 and self.seq[pos_seq].check_sat(s[0], s[1]) and self.check_sat(pos_seq + 1, s[2:])
    #         else:  # tUnion and REGEX
    #             for i in range(len(s) + 1):
    #                 ret = self.seq[pos_seq].check_sat(s[:i]) and self.check_sat(pos_seq + 1, s[i:])
    #                 if ret:
    #                     break
    #
    #         self.cache_check_sat[(pos_seq, s)] = ret
    #         return ret


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


class Composition:
    def __init__(self, *args):
        self.p = args  # reversed
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
