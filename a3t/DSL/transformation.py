from abc import ABC, abstractmethod
import os
from nltk import pos_tag
import gzip

from a3t.dataset.download import download_ppdb, download_artifacts


class Transformation(ABC):
    def __init__(self, length_preserving=False):
        """
        A default init function.
        """
        super().__init__()
        self.length_preserving = length_preserving

    @abstractmethod
    def get_pos(self, ipt):
        # get matched positions in input
        pass

    @abstractmethod
    def transformer(self, ipt, start_pos, end_pos):
        # transformer for a segment of input
        pass

    def sub_transformer(self, ipt, start_pos, end_pos):
        # substring transformer for length preserving transformation
        assert self.length_preserving


class Sub(Transformation):
    def __init__(self, use_fewer_sub, dataset_home="/tmp/.A3T", ppdb_type="s"):
        """
        Init Sub transformation
        Sub substitutes one word with its synonym with a requirement that they have the same pos_tag.
        :param use_fewer_sub: if True, one word can only be substituted by one of its synonyms.
        :param dataset_home: the home of the dataset
        :param ppdb_type the type of the ppdb synonyms can be ["s", "m", "l", "xl", "xxl", "xxxl"]
        """
        assert ppdb_type in ["s", "m", "l", "xl", "xxl", "xxxl"]
        self.use_fewer_sub = use_fewer_sub
        self.synonym_dict = {}
        self.synonym_dict_pos_tag = {}
        pddb_path = os.path.join(dataset_home, "ppdb")
        if not os.path.exists(pddb_path):
            os.mkdir(pddb_path)
        pddb_file = os.path.join(pddb_path, "ppdb-2.0-%s-lexical" % ppdb_type)
        if not os.path.exists(pddb_file):
            download_ppdb(pddb_path, ppdb_type)

        with gzip.open(pddb_file, 'rb') as f:
            lines = f.readlines()
            for line in lines:
                line = line.decode()
                tmp = line.strip().split(" ||| ")
                postag, x, y = tmp[0][1:-1], tmp[1], tmp[2]
                self.synonym_dict_add_str(x, y, postag)
                self.synonym_dict_add_str(y, x, postag)

        print("Compute synonym_dict success!")
        super(Sub, self).__init__(True)

    def synonym_dict_add_str(self, x, y, postag):
        if x not in self.synonym_dict:
            self.synonym_dict[x] = [y]
            self.synonym_dict_pos_tag[x] = [postag]
        elif not self.use_fewer_sub:
            self.synonym_dict[x].append(y)
            self.synonym_dict_pos_tag[x].append(postag)

    def get_pos(self, ipt):
        ipt_pos_tag = pos_tag(ipt)
        ret = []
        for (start_pos, (x, pos_tagging)) in enumerate(ipt_pos_tag):
            if x in self.synonym_dict:
                if any(pos_tagging == t for t in self.synonym_dict_pos_tag[x]):
                    ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        ipt_pos_tag = pos_tag(ipt)
        x = ipt[start_pos]
        pos_tagging = ipt_pos_tag[start_pos][1]
        for (w, t) in zip(self.synonym_dict[x], self.synonym_dict_pos_tag[x]):
            if t == pos_tagging:
                new_ipt = ipt[:start_pos] + [w] + ipt[end_pos:]
                yield new_ipt

    def sub_transformer(self, ipt, start_pos, end_pos):
        ipt_pos_tag = pos_tag(ipt)
        x = ipt[start_pos]
        pos_tagging = ipt_pos_tag[start_pos][1]
        for (w, t) in zip(self.synonym_dict[x], self.synonym_dict_pos_tag[x]):
            if t == pos_tagging:
                yield [w]


class SubChar(Transformation):
    def __init__(self, use_fewer_sub, dataset_home="/tmp/.A3T"):
        """
        Init SubChar transformation
        Sub substitutes one char with other chars. The substitution relation is one direction (x => y, but not x <= y)
        :param sub_file: the character substitution file
        :param use_fewer_sub: if True, one char can only be substituted by one of its synonyms.
        """
        download_artifacts(dataset_home)
        a3t_sst2test_file = os.path.join(dataset_home, "A3T-artifacts", "dataset", "en.key")
        lines = open(a3t_sst2test_file).readlines()
        self.synonym_dict = {}
        for line in lines:
            tmp = line.strip().split()
            x, y = tmp[0], tmp[1]
            if x not in self.synonym_dict:
                self.synonym_dict[x] = [y]
            elif not use_fewer_sub:
                self.synonym_dict[x].append(y)

        print("Compute set success!")
        super(SubChar, self).__init__(True)

    def get_pos(self, ipt):
        ret = []
        for (start_pos, x) in enumerate(ipt):
            if x in self.synonym_dict:
                ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        x = ipt[start_pos]
        for w in self.synonym_dict[x]:
            new_ipt = ipt[:start_pos] + [w] + ipt[end_pos:]
            yield new_ipt

    def sub_transformer(self, ipt, start_pos, end_pos):
        x = ipt[start_pos]
        for w in self.synonym_dict[x]:
            yield [w]


class Del(Transformation):
    def __init__(self, stop_words=None):
        """
        Init Del transformation
        Del removes one stop word.
        :param stop_words: stop words to delete
        """
        if stop_words is None:
            stop_words = {"a", "and", "the", "of", "to"}
        self.stop_words = stop_words

        super(Del, self).__init__()

    def get_pos(self, ipt):
        ret = []
        for (start_pos, x) in enumerate(ipt):
            if x in self.stop_words:
                ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        new_ipt = ipt[:start_pos] + ipt[end_pos:]
        yield new_ipt


class Ins(Transformation):
    def __init__(self):
        """
        Init Ins transformation
        Ins duplicates a word behind it.
        """
        super(Ins, self).__init__()

    def get_pos(self, ipt):
        return [(x, x + 1) for x in range(len(ipt))]

    def transformer(self, ipt, start_pos, end_pos):
        new_ipt = ipt[:start_pos] + [ipt[start_pos], ipt[start_pos]] + ipt[end_pos:]
        yield new_ipt


class InsChar(Transformation):
    def __init__(self, use_fewer_sub, dataset_home="/tmp/.A3T"):
        """
        Init SubChar transformation
        Sub substitutes one char with other chars. The substitution relation is one direction (x => y, but not x <= y)
        :param sub_file: the character substitution file
        :param use_fewer_sub: if True, one char can only be substituted by one of its synonyms.
        """
        download_artifacts(dataset_home)
        a3t_sst2test_file = os.path.join(dataset_home, "A3T-artifacts", "dataset", "en.key")
        lines = open(a3t_sst2test_file).readlines()
        self.synonym_dict = {}
        for line in lines:
            tmp = line.strip().split()
            x, y = tmp[0], tmp[1]
            if x not in self.synonym_dict:
                self.synonym_dict[x] = [y]
            elif not use_fewer_sub:
                self.synonym_dict[x].append(y)

        print("Compute set success!")
        super(InsChar, self).__init__()

    def get_pos(self, ipt):
        ret = []
        for (start_pos, x) in enumerate(ipt):
            if x in self.synonym_dict:
                ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        x = ipt[start_pos]
        for w in self.synonym_dict[x]:
            new_ipt = ipt[:start_pos] + [x, w] + ipt[end_pos:]
            yield new_ipt


class Swap(Transformation):
    def __init__(self):
        """
        Init Swap transformation
        Swap two adjacent tokens.
        """
        super(Swap, self).__init__(True)

    def get_pos(self, ipt):
        return [(x, x + 2) for x in range(len(ipt) - 1)]

    def transformer(self, ipt, start_pos, end_pos):
        new_ipt = ipt[:start_pos] + [ipt[start_pos + 1], ipt[start_pos]] + ipt[end_pos:]
        yield new_ipt

    def sub_transformer(self, ipt, start_pos, end_pos):
        yield [ipt[start_pos + 1], ipt[start_pos]]
