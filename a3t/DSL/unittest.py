import numpy as np
import unittest
import random
import itertools

from a3t.DSL.general_HotFlip import GeneralHotFlipAttack, Candidate
from a3t.DSL.transformation import Sub, Ins, Del
from utils import Beam


class TestModel1:
    def __init__(self, dim):
        self.embed = {}
        self.dim = dim


class TestModel:
    def __init__(self, dim):
        self.embed = {}
        self.dim = dim

    def get_grad(self, x, y):
        self.scan(x)
        return np.array([self.embed[w] for w in x])

    def get_embed(self, x, ret_len=None):
        self.scan(x)
        if ret_len is None:
            ret_len = len(x)
        ret = np.zeros((ret_len, self.dim))
        for (i, w) in enumerate(x):
            ret[i] = self.embed[w]
        return ret

    def scan(self, x):
        for w in x:
            if w not in self.embed:
                self.embed[w] = np.random.random(self.dim)


class TestGeneralHotFlipAttack(unittest.TestCase):
    def test_arguments(self):
        with self.assertRaises(AttributeError):
            GeneralHotFlipAttack([(1, 3, 4)])
        with self.assertRaises(AttributeError):
            GeneralHotFlipAttack([(1, 2)])
        attack = GeneralHotFlipAttack([(Sub("dataset", True), 1), (Del(), 2), (Ins(), 3)])
        with self.assertRaises(AttributeError):
            attack.gen_adv(TestModel1(100), [], 1, 5, None)

    def test_beam(self):
        beams = [Beam(4), Beam(3), Beam(2)]
        test_sets = [[("aaa", 1), ("bbb", 0), ("ccc", -3)], [("aaa", 1), ("bbb", -3), ("ccc", 0)],
                     [("aaa", 1), ("bbb", -3), ("ccc", 0), ("ddd", 2)]]
        for (candidates, tests) in zip(beams, test_sets):
            random.shuffle(tests)
            for a in tests:
                candidates.add(*a)
            tests.sort(key=lambda x: -x[1])
            tests = tests[:candidates.budget]
            ans = candidates.check_balance()
            self.assertEqual(len(tests), len(ans))
            for (a, b) in zip(tests, ans):
                self.assertTupleEqual(a, b)

    def test_perturbation(self):
        model = TestModel(100)
        perturb = [(Sub("dataset", True), 3), (Del(), 2), (Ins(), 1)]
        attack = GeneralHotFlipAttack(perturb)
        sen = "i see that a cat sits on the floor .".split()
        sub_pos = perturb[0][0].get_pos(sen)
        del_pos = perturb[1][0].get_pos(sen)
        ins_pos = perturb[2][0].get_pos(sen)
        cnt = 0
        for sub_cnt in range(perturb[0][1] + 1):
            for del_cnt in range(perturb[1][1] + 1):
                for ins_cnt in range(perturb[2][1] + 1):
                    for a in itertools.permutations(sub_pos, sub_cnt):
                        for b in itertools.permutations(del_pos, del_cnt):
                            for c in itertools.permutations(ins_pos, ins_cnt):
                                a = set(a)
                                b = set(b)
                                c = set(c)
                                if a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c):
                                    cnt += 1

        ans = attack.gen_adv(model, sen, 0, 10000, model.get_embed, True)
        self.assertEqual(cnt, len(ans))
        for i in range(1, len(ans)):
            self.assertTrue(ans[i][1] <= ans[i - 1][1])

    def test_multi_choice(self):
        model = TestModel(100)
        perturb = [(Sub("dataset", False), 3)]
        attack = GeneralHotFlipAttack(perturb)
        sen = "i see that a cat sits on the floor .".split()
        sub_pos = perturb[0][0].get_pos(sen)
        for i, (start_pos, end_pos) in enumerate(sub_pos):
            cnt = 0
            for _ in perturb[0][0].transformer(sen, start_pos, end_pos):
                cnt += 1
            sub_pos[i] = cnt

        cnt = 0
        for sub_cnt in range(perturb[0][1] + 1):
            for a in itertools.permutations(sub_pos, sub_cnt):
                cnt += np.prod(list(a))

        ans = attack.gen_adv(model, sen, 0, 10000, model.get_embed, True)
        self.assertEqual(cnt, len(ans))
        for i in range(1, len(ans)):
            self.assertTrue(ans[i][1] <= ans[i - 1][1])

    def test_single_transformation(self):
        model = TestModel(100)
        perturb1 = [(Sub("dataset", True), 2)]
        perturb2 = [(Del(), 2)]
        perturb3 = [(Ins(), 2)]
        attack1 = GeneralHotFlipAttack(perturb1)
        attack2 = GeneralHotFlipAttack(perturb2)
        attack3 = GeneralHotFlipAttack(perturb3)
        sen = "i see that a cat sits on the floor .".split()
        sub_pos = perturb1[0][0].get_pos(sen)
        exp_ans1 = []
        for sub_cnt in range(perturb1[0][1] + 1):
            for a in itertools.permutations(sub_pos, sub_cnt):
                new_x = sen
                for (start_pos, end_pos) in a:
                    new_x = next(perturb1[0][0].transformer(new_x, start_pos, end_pos))
                exp_ans1.append(new_x)

        del_pos = perturb2[0][0].get_pos(sen)
        exp_ans2 = []
        for del_cnt in range(perturb2[0][1] + 1):
            for a in itertools.permutations(del_pos, del_cnt):
                new_x = []
                for (i, w) in enumerate(sen):
                    if (i, i + 1) not in a:
                        new_x.append(w)

                exp_ans2.append(new_x)

        ins_pos = perturb3[0][0].get_pos(sen)
        exp_ans3 = []
        for ins_cnt in range(perturb3[0][1] + 1):
            for a in itertools.permutations(ins_pos, ins_cnt):
                new_x = []
                for (i, w) in enumerate(sen):
                    if (i, i + 1) not in a:
                        new_x.append(w)
                    else:
                        new_x.append(w)
                        new_x.append(w)

                exp_ans3.append(new_x)

        ans1 = attack1.gen_adv(model, sen, 0, 10000, model.get_embed)
        ans2 = attack2.gen_adv(model, sen, 0, 10000, model.get_embed)
        ans3 = attack3.gen_adv(model, sen, 0, 10000, model.get_embed)

        def list2map(l):
            ret = {}
            for x in l:
                x = tuple(x)
                ret[x] = ret.setdefault(x, 0) + 1
            return ret

        self.assertDictEqual(list2map(exp_ans1), list2map(ans1))
        self.assertDictEqual(list2map(exp_ans2), list2map(ans2))
        self.assertDictEqual(list2map(exp_ans3), list2map(ans3))

    def test_map_ori2x(self):
        model = TestModel(100)
        perturb = [(Sub("dataset", True), 3), (Del({"floor"}), 2), (Ins(), 1)]
        sen = "i see that a cat sits on the floor .".split()
        sub_pos = perturb[0][0].get_pos(sen)
        del_pos = perturb[1][0].get_pos(sen)
        ins_pos = perturb[2][0].get_pos(sen)
        candidate = Candidate(["dummy"], 0, [None] * len(sen))
        candidates = Beam(1)
        candidate.try_all_pos(ins_pos, perturb[2][0], model.get_grad(candidate.x, 0), model.get_embed,
                              candidates)
        self.assertEqual(0, len(candidates.queue))
        map_ori2x = [None] * len(sen)
        map_ori2x[sen.index("floor")] = 1
        candidate = Candidate(["dummy1", "floor", "dummy2"], 0, map_ori2x)  # all perturbation can be applied
        candidates = Beam(1)
        candidate.try_all_pos(sub_pos, perturb[0][0], model.get_grad(candidate.x, 0), model.get_embed,
                              candidates)
        self.assertSequenceEqual(["dummy1", "flooring", "dummy2"], candidates.check_balance()[0][0].x)

        candidates = Beam(1)
        candidate.try_all_pos(del_pos, perturb[1][0], model.get_grad(candidate.x, 0), model.get_embed,
                              candidates)
        self.assertSequenceEqual(["dummy1", "dummy2"], candidates.check_balance()[0][0].x)

        candidates = Beam(1)
        candidate.try_all_pos(ins_pos, perturb[2][0], model.get_grad(candidate.x, 0), model.get_embed,
                              candidates)
        self.assertSequenceEqual(["dummy1", "floor", "floor", "dummy2"], candidates.check_balance()[0][0].x)


if __name__ == '__main__':
    unittest.main()
