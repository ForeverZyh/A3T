import a3t.diffai.components as n


def WordLevelSST2(glove, c=2, fst_conv_window=5, **kargs):
    return n.Seq(n.Embedding(glove=glove, span=fst_conv_window * 2), n.Conv4Embed(100, fst_conv_window, bias=True),
                 n.AvgPool2D4Embed(fst_conv_window), n.ReduceToZono(),
                 n.FFNN([c], last_lin=True, last_zono=True, **kargs))


def CharLevelSST2(vocab, dim, c=2, fst_conv_window=5, **kargs):
    return n.Seq(n.Embedding(vocab=vocab, dim=dim, span=fst_conv_window * 2),
                 n.Conv4Embed(100, fst_conv_window, bias=True), n.AvgPool2D4Embed(fst_conv_window), n.ReduceToZono(),
                 n.FFNN([c], last_lin=True, last_zono=True, **kargs))


def CharLevelAG(vocab, dim, c=4, fst_conv_window=10, **kargs):
    return n.Seq(n.Embedding(vocab=vocab, dim=dim, span=fst_conv_window * 2),
                 n.Conv4Embed(64, fst_conv_window, bias=True), n.AvgPool2D4Embed(fst_conv_window), n.ReduceToZono(),
                 n.FFNN([64, 64, c], last_lin=True, last_zono=True, **kargs))
