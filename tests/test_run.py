from a3t.diffai.models import WordLevelSST2
from a3t.dataset.dataset_loader import SST2WordLevel, SST2CharLevel, Glove
from a3t.diffai.helpers import loadDataset
from a3t.DSL.transformation import Sub, SubChar, Del, Ins, InsChar, Swap


def prepare_SST2WordLevel():
    Glove.build(6, 50)
    SST2WordLevel.build()


def prepare_SST2CharLevel():
    SST2CharLevel.build()


# load data
prepare_SST2WordLevel()
prepare_SST2CharLevel()
batch_size = 32
sst2word_loader_train = loadDataset(SST2WordLevel, batch_size, "train")
sst2word_loader_val = loadDataset(SST2WordLevel, batch_size, "val")
sst2word_loader_test = loadDataset(SST2WordLevel, batch_size, "test")
sst2char_loader_train = loadDataset(SST2CharLevel, batch_size, "train")
sst2char_loader_val = loadDataset(SST2CharLevel, batch_size, "val")
sst2char_loader_test = loadDataset(SST2CharLevel, batch_size, "test")

# transformations
word_perturbation = [(Sub(True), 2), (Ins(), 2), (Del(), 2)]
char_perturbation = [(SubChar(True), 2), (InsChar(True), 2), (Del(set()), 2), (Swap(), 2)]
# print(word_perturbation[0][0].synonym_dict)
# print(char_perturbation[0][0].synonym_dict)
