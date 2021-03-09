import torch
from torch import nn
import math

import a3t.diffai.helpers as h
import a3t.diffai.ai as ai


class Const():
    def __init__(self, c):
        self.c = c if c is None else float(c)

    def getVal(self, c=None, **kargs):
        return self.c if self.c is not None else c

    def __str__(self):
        return str(self.c)

    def initConst(x):
        return x if isinstance(x, Const) else Const(x)


class Lin(Const):
    def __init__(self, start, end, steps, initial=0, quant=False):
        self.start = float(start)
        self.end = float(end)
        self.steps = float(steps)
        self.initial = float(initial)
        self.quant = quant

    def getVal(self, time=0, **kargs):
        if self.quant:
            time = math.floor(time)
        return (self.end - self.start) * max(0, min(1, float(time - self.initial) / self.steps)) + self.start

    def __str__(self):
        return "Lin(%s,%s,%s,%s, quant=%s)".format(str(self.start), str(self.end), str(self.steps), str(self.initial),
                                                   str(self.quant))


class Until(Const):
    def __init__(self, thresh, a, b):
        self.a = Const.initConst(a)
        self.b = Const.initConst(b)
        self.thresh = thresh

    def getVal(self, *args, time=0, **kargs):
        return self.a.getVal(*args, time=time, **kargs) if time < self.thresh else self.b.getVal(*args,
                                                                                                 time=time - self.thresh,
                                                                                                 **kargs)

    def __str__(self):
        return "Until(%s, %s, %s)" % (str(self.thresh), str(self.a), str(self.b))


class Scale(Const):  # use with mix when aw = 1, and 0 <= c < 1
    def __init__(self, c):
        self.c = Const.initConst(c)

    def getVal(self, *args, **kargs):
        c = self.c.getVal(*args, **kargs)
        if c == 0:
            return 0
        assert c >= 0
        assert c < 1
        return c / (1 - c)

    def __str__(self):
        return "Scale(%s)" % str(self.c)


def MixLin(*args, **kargs):
    return Scale(Lin(*args, **kargs))


class Normal(Const):
    def __init__(self, c):
        self.c = Const.initConst(c)

    def getVal(self, *args, shape=[1], **kargs):
        c = self.c.getVal(*args, shape=shape, **kargs)
        return torch.randn(shape, device=h.device).abs() * c

    def __str__(self):
        return "Normal(%s)" % str(self.c)


class Clip(Const):
    def __init__(self, c, l, u):
        self.c = Const.initConst(c)
        self.l = Const.initConst(l)
        self.u = Const.initConst(u)

    def getVal(self, *args, **kargs):
        c = self.c.getVal(*args, **kargs)
        l = self.l.getVal(*args, **kargs)
        u = self.u.getVal(*args, **kargs)
        if isinstance(c, float):
            return min(max(c, l), u)
        else:
            return c.clamp(l, u)

    def __str__(self):
        return "Clip(%s, %s, %s)" % (str(self.c), str(self.l), str(self.u))


class Fun(Const):
    def __init__(self, foo):
        self.foo = foo

    def getVal(self, *args, **kargs):
        return self.foo(*args, **kargs)

    def __str__(self):
        return "Fun(...)"


class Complement(Const):  # use with mix when aw = 1, and 0 <= c < 1
    def __init__(self, c):
        self.c = Const.initConst(c)

    def getVal(self, *args, **kargs):
        c = self.c.getVal(*args, **kargs)
        assert c >= 0
        assert c <= 1
        return 1 - c

    def __str__(self):
        return "Complement(%s)" % str(self.c)


class DataParallelAI(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelAI, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        def inner_scatter(inputs, target_gpus, dim=0):
            if isinstance(inputs, tuple) and len(inputs) == 1:
                inputs = inputs[0]
            if isinstance(inputs, ai.ListDomain):
                rets = [type(inputs)([]) for _ in range(len(device_ids))]
                for (i, a) in enumerate(inputs.al):
                    tmp_rets = inner_scatter(a, target_gpus, dim)
                    assert len(tmp_rets) == len(rets)
                    for ret, tmp_ret in zip(rets, tmp_rets):
                        ret.al.append(tmp_ret)
                return rets
            elif isinstance(inputs, ai.TaggedDomain):
                tmp_rets = inner_scatter(inputs.a, target_gpus, dim)
                rets = []
                for tmp_ret in tmp_rets:
                    rets.append(type(inputs)(tmp_ret, inputs.tag))
                return rets
            elif isinstance(inputs, ai.LabeledDomain):
                tmp_rets = inner_scatter(inputs.o, target_gpus, dim)
                rets = []
                for tmp_ret in tmp_rets:
                    rets.append(type(inputs)(inputs.label))
                    rets[-1].box(tmp_ret)
                return rets
            elif isinstance(inputs, ai.HybridZonotope):
                head = inner_scatter(inputs.head, target_gpus, dim)
                errors = None if inputs.errors is None else inner_scatter(inputs.errors, target_gpus, 1)
                beta = None if inputs.beta is None else inner_scatter(inputs.beta, target_gpus, dim)
                rets = []
                for i in range(len(head)):
                    rets.append(type(inputs)(head[i], beta[i] if beta is not None else None,
                                             errors[i] if errors is not None else None))
                return rets
            else:
                return nn.parallel.scatter_gather.scatter(inputs, target_gpus, dim)

        def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
            r"""Scatter with support for kwargs dictionary"""
            inputs = inner_scatter(inputs, target_gpus, dim) if inputs else []
            kwargs = inner_scatter(kwargs, target_gpus, dim) if kwargs else []
            if len(inputs) < len(kwargs):
                inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
            elif len(kwargs) < len(inputs):
                kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
            inputs = tuple(inputs)
            kwargs = tuple(kwargs)
            return inputs, kwargs

        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, target_device, dim=0):
        def inner_gather(outputs, target_gpus, dim=0):
            if isinstance(outputs[0], ai.ListDomain):
                ret = type(outputs[0])([])
                for i in range(len(outputs[0].al)):
                    t = [o.al[i] for o in outputs]
                    tmp_rets = inner_gather(t, target_gpus, dim)
                    ret.al.append(tmp_rets)

                return ret
            elif isinstance(outputs[0], ai.TaggedDomain):
                t = [o.a for o in outputs]
                tmp_rets = inner_gather(t, target_gpus, dim)
                return type(outputs[0])(tmp_rets, outputs[0].tag)
            elif isinstance(outputs[0], ai.LabeledDomain):
                t = [ot.o for ot in outputs]
                tmp_rets = inner_gather(t, target_gpus, dim)
                ret = type(outputs[0])(outputs[0].label)
                ret.box(tmp_rets)
                return ret
            elif isinstance(outputs[0], ai.HybridZonotope):
                head = inner_gather([o.head for o in outputs], target_gpus, dim)
                errors = None if outputs[0].errors is None else inner_gather([o.errors for o in outputs], target_gpus,
                                                                             1)
                beta = None if outputs[0].beta is None else inner_gather([o.beta for o in outputs], target_gpus, dim)
                return type(outputs[0])(head, beta, errors)
            else:
                return nn.parallel.scatter_gather.gather(outputs, target_gpus, dim)

        return inner_gather(outputs, target_device, dim)


class Info:
    out_y = None
    adv = False
    adjacent_keys = None
