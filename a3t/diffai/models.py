import torch
import numpy as np
import torch.nn as nn

import a3t.diffai.components as n
import a3t.diffai.scheduling as S
import a3t.diffai.goals as goals
from a3t.dataset.dataset_loader import Glove, SST2CharLevel


class ModelWrapper:
    def __init__(self, model, vocab, device, vargs, pad_to=None):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.vargs = vargs
        self.pad_to = pad_to
        for layer in model.net.layers:
            if isinstance(layer, n.Embedding):
                self.embed_layer = layer
                break

    def to_strs(self, ids):
        end = len(ids)
        while end > 0 and ids[end - 1] == 0:
            end -= 1
        return [self.vocab.id2str[i] for i in ids[:end]]

    def to_ids(self, s):
        ret = [self.vocab.str2id[c] for c in s]
        if self.pad_to is None:
            return ret
        else:
            return ret + [0] * (self.pad_to - len(ret))

    def get_embed(self, x, ret_len=None):
        """
        :param x: a list of tokens
        :param ret_len: the return length, if None, the length is equal to len(x). if len(x) < ret_len, we add padding
        :return: np array with shape (ret_len, word_vec_size)
        """
        x = torch.LongTensor([self.vocab.get_index(w) for w in x]).to(self.device)
        if ret_len is None:
            ret_len = len(x)
        ret = np.zeros((ret_len, self.embed_layer.dim))
        with torch.no_grad():
            ret[:len(x)] = self.embed_layer.embed(x.unsqueeze(0))[0].cpu().numpy()
        return ret

    def get_grad(self, x, y):
        """
        :param x: a list of tokens
        :param y: a label
        :return: np array with shape (len, word_vec_size)
        """
        pre_len = len(x)
        x = torch.LongTensor(self.to_ids(x)).to(self.device)
        gradients = self.partial_to_loss(x.view(1, -1), torch.LongTensor([y]).to(self.device))[:pre_len]
        return gradients

    def partial_to_loss(self, x, y):
        self.model.eval()
        assert not self.model.training
        S.TrainInfo.adv = True
        self.model.optimizer.zero_grad()
        loss = self.model.aiLoss(x, y, **self.vargs).mean(dim=0)
        loss.backward()
        self.model.train()
        S.TrainInfo.adv = False
        return S.TrainInfo.out_y.grad[0][0].cpu().numpy()


def WordLevelSST2(c, fst_conv_window=5, **kargs):
    return n.Seq(n.Embedding(glove=Glove, span=fst_conv_window * 2), n.Conv4Embed(100, fst_conv_window, bias=True),
                 n.AvgPool2D4Embed(fst_conv_window), n.ReduceToZono(),
                 n.FFNN([c], last_lin=True, last_zono=True, **kargs))


def CharLevelSST2(c, dim=150, fst_conv_window=5, **kargs):
    return n.Seq(n.Embedding(vocab=len(SST2CharLevel.id2str), dim=dim, span=fst_conv_window * 2),
                 n.Conv4Embed(100, fst_conv_window, bias=True), n.AvgPool2D4Embed(fst_conv_window), n.ReduceToZono(),
                 n.FFNN([c], last_lin=True, last_zono=True, **kargs))


# def CharLevelAG(vocab, dim=64, c=4, fst_conv_window=10, **kargs):
#     return n.Seq(n.Embedding(vocab=vocab, dim=dim, span=fst_conv_window * 2),
#                  n.Conv4Embed(64, fst_conv_window, bias=True), n.AvgPool2D4Embed(fst_conv_window), n.ReduceToZono(),
#                  n.FFNN([64, 64, c], last_lin=True, last_zono=True, **kargs))


class Top(nn.Module):
    def __init__(self, args, net, ty=goals.Point):
        super(Top, self).__init__()
        self.net = net
        self.ty = ty
        self.w = args.width
        self.global_num = 0
        self.speedCount = 0
        self.speed = 0.0
        self.regularize = args.regularize

    def addSpeed(self, s):
        self.speed = (s + self.speed * self.speedCount) / (self.speedCount + 1)
        self.speedCount += 1

    def forward(self, x):
        return self.net(x)

    def clip_norm(self):
        self.net.clip_norm()

    def boxSpec(self, x, target, **kargs):
        return [(self.ty.box(x, w=self.w, model=self, target=target, untargeted=True, **kargs).to_dtype(), target)]

    def regLoss(self):
        if self.regularize is None or self.regularize <= 0.0:
            return 0
        r = self.net.regularize(2)
        return self.regularize * r

    def aiLoss(self, dom, target, **args):
        r = self(dom)
        return self.regLoss() + r.loss(target=target, **args)

    def printNet(self, f):
        self.net.printNet(f)
