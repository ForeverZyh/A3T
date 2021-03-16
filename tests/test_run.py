from a3t.diffai.train import train
from a3t.dataset.dataset_loader import SST2WordLevel, SST2CharLevel, Glove
from a3t.diffai.helpers import loadDataset
from a3t.DSL.transformation import Sub, SubChar, Del, Ins, InsChar, Swap


def prepare_SST2WordLevel():
    Glove.build(6, 50)
    SST2WordLevel.build()


def prepare_SST2CharLevel():
    SST2CharLevel.build()


class Args:
    batch_size = 10
    model_srt = ""
    clip_norm = False
    epochs = 20
    log_freq = 10
    save_freq = 1
    lr = 1e-3
    threshold = -0.01
    lr_patience = 0
    early_stop_patience = 5
    factor = 0.5
    max_norm = 10000
    width = 0.01
    out = "out"
    adv_train_num = 2
    adv_test_num = 0
    epoch_perct_decay = 0.6
    train_lambda = 0.75
    regularize = 0.005
    seed = 12345


# load data

prepare_SST2WordLevel()
sst2word_loader_train = loadDataset(SST2WordLevel, Args.batch_size, "train", test_slice=slice(0, 3000, None))
sst2word_loader_val = loadDataset(SST2WordLevel, Args.batch_size, "val", test_slice=slice(0, 100, None))
sst2word_loader_test = loadDataset(SST2WordLevel, Args.batch_size, "test", test_slice=slice(0, 100, None))

prepare_SST2CharLevel()
sst2char_loader_train = loadDataset(SST2CharLevel, Args.batch_size, "train", test_slice=slice(0, 30, None))
sst2char_loader_val = loadDataset(SST2CharLevel, Args.batch_size, "val", test_slice=slice(0, 10, None))
sst2char_loader_test = loadDataset(SST2CharLevel, Args.batch_size, "test", test_slice=slice(0, 10, None))

# transformations
word_perturbation = [(Sub(True), 2), (Ins(), 2), (Del(), 2)]
char_perturbation = [(SubChar(True), 2), (InsChar(True), 2), (Del(set()), 2), (Swap(), 2)]
# print(word_perturbation[0][0].synonym_dict)
# print(char_perturbation[0][0].synonym_dict)

Args.model_srt = "CharLevelSST2"
# Train
train(SST2CharLevel.make_vocab(), sst2char_loader_train, sst2char_loader_val, sst2char_loader_test,
      [(InsChar(True), 2), (Del(set()), 2)], [(SubChar(True), 2), (Swap(), 2)], fixed_len=SST2CharLevel.max_len,
      args=Args)

# # Test
# train(SST2CharLevel.make_vocab(), sst2char_loader_train, sst2char_loader_val, sst2char_loader_test,
#       [(InsChar(True), 2), (Del(set()), 2)], [(SubChar(True), 2), (Swap(), 2)], fixed_len=SST2CharLevel.max_len,
#       args=Args, test=True,
#       load_path="out/XXX.pynet")


Args.model_srt = "WordLevelSST2"
# Train
train(SST2WordLevel.vocab, sst2word_loader_train, sst2word_loader_val, sst2word_loader_test,
      [(Ins(), 2), (Del(), 2)], [(Sub(True), 2)], fixed_len=SST2WordLevel.max_len, args=Args)

# # Test
# train(SST2WordLevel.vocab, sst2word_loader_train, sst2word_loader_val, sst2word_loader_test,
#       [(Ins(), 2), (Del(), 2)], [(Sub(True), 2)], fixed_len=SST2WordLevel.max_len, args=Args,
#       test=True,
#       load_path="out/YYY.pynet")
