import numpy as np

from a3t.DSL.transformation import Transformation
from utils import Beam


class GeneralHotFlipAttack:
    def __init__(self, perturbation: list, use_random_aug=False):
        """
        :param list perturbation: A perturbation space specified in the DSL.
        For example, [(Sub(), 2), (Del(), 1)] means at most 2 Sub string transformations and at most 1 Del string
        transformation. Sub and Del are default string transformations (see transformation.py).

        :Classifier Capacity: Gradient

        The idea of writing a perturbation space in a DSL is first proposed and implemented in the following paper:
        Robustness to Programmable String Transformations via Augmented Abstract Training.
        Yuhao Zhang, Aws Albarghouthi, Loris D’Antoni.
        `[pdf] <https://arxiv.org/abs/2002.09579>`__
        `[code] <https://github.com/ForeverZyh/A3T>`__

        """

        try:
            for (a, b) in perturbation:
                assert isinstance(a, Transformation)
                assert isinstance(b, int)
        except:
            raise AttributeError("param perturbation %s is not in the correct form." % str(perturbation))

        self.perturbation = perturbation
        self.use_random_aug = use_random_aug

    def gen_adv(self, model, x: list, y, top_n: int, get_embed, return_score=False):
        """
        Beam search for the perturbation space. The order of beam search is the same in the perturbation DSL.
        Some adversarial attack tries to rearrange the order of beam search for better effectiveness.
        TODO: the order of beam search can be rearranged for better effectiveness.
        :param model: the victim model, which has to support a method get_grad.
        :param x: a list of input tokens.
        :param y: the correct label of input x.
        :param top_n: maximum number of adversarial candidates given the perturbation space.
        :param get_embed: get_embed(x, ret_len) takes a list of tokens as inputs and output the embedding matrix with
        shape (ret_len, dim).
        if ret_len > len(x), then padding is needed.
        if ret_len < len(x), then truncating is needed. (Currently, we do not need truncate)
        :param return_score: whether return the score as a list [(sen, score)], default False, i.e., return [sen]
        :return: a list of adversarial examples
        """

        try:
            model.get_grad
        except AttributeError:
            raise AttributeError("The victim model does not support get_grad method.")

        candidate = Candidate(x, 0 if not self.use_random_aug else np.random.random())
        candidates = Beam(top_n)
        candidates.add(candidate, candidate.score)
        for (tran, delta) in self.perturbation:
            possible_pos = tran.get_pos(x)  # get a list of possible positions
            perturbed_set = set()  # restore the perturbed candidates to eliminate duplications
            for _ in range(delta):
                # keep the old candidates because we will change candidates in the following loop
                old_candidates = candidates.check_balance()
                for (candidate, _) in old_candidates:
                    if candidate not in perturbed_set:
                        if len(candidate.x) > 0:
                            if self.use_random_aug:
                                candidate.try_all_pos(possible_pos, tran, None, None, candidates)
                            else:
                                candidate.try_all_pos(possible_pos, tran, model.get_grad(candidate.x, y), get_embed,
                                                      candidates)
                        perturbed_set.add(candidate)

        ret = candidates.check_balance()
        if return_score:
            return [(x.x, x.score) for (x, _) in ret]
        else:
            return [x.x for (x, _) in ret]


class Candidate:
    def __init__(self, x, score, map_ori2x=None):
        """
        Init a candidate
        :param x: input tokens
        :param score: score of adversarial attack, the larger the better
        :param map_ori2x: position mapping from the original input to transformed input
        """
        self.x = x
        self.score = score
        if map_ori2x is None:
            self.map_ori2x = [_ for _ in range(len(x))]
        else:
            self.map_ori2x = map_ori2x

    def __lt__(self, other):
        return self.x < other.x

    def try_all_pos(self, pos: list, tran: Transformation, gradients, get_embed, candidates: Beam):
        """
        Try all possible positions for trans
        :param pos: possible positions, a list of (start_pos, end_pos)
        :param tran: the target transformation
        :param gradients: the gradients tensor with respect to self.x
        :param get_embed: a function for getting the embedding of a list of tokens
        :param candidates: a beam of candidates, will be modified by this methods
        :return: None
        """

        for (start_pos_ori, end_pos_ori) in pos:
            if all(self.map_ori2x[i] is not None for i in range(start_pos_ori, end_pos_ori)):
                start_pos_x = self.map_ori2x[start_pos_ori]
                # notice that self.map_ori2x[end_pos] can be None, we need to calculate from self.map_ori2x[end_pos - 1]
                end_pos_x = self.map_ori2x[end_pos_ori - 1] + 1

                for new_x in tran.transformer(self.x, start_pos_x, end_pos_x):
                    delta_len = len(new_x) - len(self.x)
                    if get_embed is not None and gradients is not None:
                        old_embedding = get_embed(self.x[start_pos_x:])
                        # ret_len specifies the ret length (padding if not enought)
                        new_embedding = get_embed(new_x[start_pos_x:min(len(new_x), len(self.x))],
                                                  ret_len=len(self.x) - start_pos_x)
                        # gradients[start_pos_x:] has shape (len(self.x) - start_pos_x, dim)
                        new_score = self.score + np.sum(gradients[start_pos_x:] * (new_embedding - old_embedding))
                    else:  # we use random sampling
                        new_score = np.random.random()
                    new_map_ori2x = self.map_ori2x[:start_pos_ori] + [None] * (end_pos_ori - start_pos_ori) + [
                        p if p is None else p + delta_len for p in self.map_ori2x[end_pos_ori:]]
                    new_candidate = Candidate(new_x, new_score, new_map_ori2x)
                    candidates.add(new_candidate, new_score)
