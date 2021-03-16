import tensorflow_datasets as tfds
from os import mkdir
from os.path import join, exists
from nltk.tokenize import word_tokenize
import numpy as np

from a3t.dataset.download import download_wordvecs, download_artifacts

DATASET_HOME = join("/tmp", ".A3T")
if not exists(DATASET_HOME):
    mkdir(DATASET_HOME)


class Vocab:
    def __init__(self, g):
        self.str2id = g.str2id
        self.id2str = g.id2str

    def get_index(self, x):
        if x in self.str2id:
            return self.str2id[x]
        return self.str2id["_UNK_"]

    def get_word(self, x):
        return self.id2str[x]


class Glove:
    str2id = {"_UNK_": 0}
    id2str = ["_UNK_"]
    embedding = []
    is_built = False

    @staticmethod
    def build(all_vocab, dim, dataset_home=DATASET_HOME):
        if Glove.is_built:  # make sure it will only be built once
            return
        Glove.is_built = True
        glove_path = join(dataset_home, "glove")
        if not exists(glove_path):
            mkdir(glove_path)
        assert (all_vocab, dim) in [(6, 50), (6, 100), (6, 200), (6, 300), (42, 300), (840, 300)]
        glove_file = join(glove_path, "glove.%dB.%dd.txt" % (all_vocab, dim))
        cached_glove_npyfile = join(glove_path, "glove.%dB.%dd.npy" % (all_vocab, dim))
        cached_glove_dict_npyfile = join(glove_path, "glove_dict.%dB.%dd.npy" % (all_vocab, dim))
        cached_glove_file = join(glove_path, "glove.%dB.%dd.npy" % (all_vocab, dim))
        cached_glove_dict_file = join(glove_path, "glove_dict.%dB.%dd.npy" % (all_vocab, dim))

        if exists(cached_glove_npyfile) and exists(cached_glove_dict_file):
            Glove.embedding = np.load(cached_glove_npyfile)
            Glove.str2id = dict(np.load(cached_glove_dict_npyfile, allow_pickle=True).item())
            Glove.id2str = [None] * len(Glove.str2id)
            for x in Glove.str2id:
                Glove.id2str[Glove.str2id[x]] = x
            print("Loading cached glove embedding success!")
            return

        if not exists(glove_file):
            download_wordvecs(glove_path, all_vocab, dim)

        print("No cached glove embedding.")
        print("Using ", glove_file, " ...")

        lines = open(glove_file).readlines()

        for (i, line) in enumerate(lines):
            tmp = line.strip().split()
            Glove.embedding.append(np.array([float(x) for x in tmp[1:]]))
            Glove.str2id[tmp[0]] = i + 1
            Glove.id2str.append(tmp[0])

        # add embedding for UNK
        Glove.embedding = [np.zeros_like(Glove.embedding[0])] + Glove.embedding

        Glove.embedding = np.array(Glove.embedding)
        np.save(cached_glove_file, Glove.embedding)
        np.save(cached_glove_dict_file, Glove.str2id)
        print("Loading glove embedding success!")

    @staticmethod
    def make_vocab():
        assert Glove.is_built
        return Vocab(Glove)


class SST2CharLevel:
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    val_X = []
    val_y = []
    max_len = 268
    is_built = False
    str2id = {"_UNK_": 0}
    id2str = ["_UNK_"]

    @staticmethod
    def map(c):
        if c not in SST2CharLevel.str2id:
            SST2CharLevel.str2id[c] = len(SST2CharLevel.str2id)
            SST2CharLevel.id2str.append(c)

        return SST2CharLevel.str2id[c]

    @staticmethod
    def build(dataset_home=DATASET_HOME):
        if SST2CharLevel.is_built:  # make sure it will only be built once
            return
        SST2CharLevel.is_built = True

        sst2char_path = join(dataset_home, "sst2char")
        if not exists(sst2char_path):
            mkdir(sst2char_path)
        download_artifacts(dataset_home)
        a3t_sst2test_file = join(dataset_home, "A3T-artifacts", "dataset", "sst2test.txt")

        def prepare_ds(ds):
            X = []
            y = []
            for features in tfds.as_numpy(ds):
                sentence, label = features["sentence"], features["label"]
                sentence = sentence.decode('UTF-8')
                x = np.zeros(SST2CharLevel.max_len, dtype=np.int)
                for (i, c) in enumerate(sentence):
                    x[i] = SST2CharLevel.map(c)
                X.append(x)
                y.append(label)

            return np.array(X), np.array(y)

        def prepare_test_ds(ds):
            X = []
            y = []
            for features in ds:
                features = features.strip()
                sentence, label = features[2:], features[:1]
                x = np.zeros(SST2CharLevel.max_len, dtype=np.int)
                for (i, c) in enumerate(sentence):
                    x[i] = SST2CharLevel.map(c)
                X.append(x)
                y.append(int(label))

            return np.array(X), np.array(y)

        ds_train = tfds.load(name="glue/sst2", split="train", shuffle_files=False)
        SST2CharLevel.train_X, SST2CharLevel.train_y = prepare_ds(ds_train)
        print("Loading training dataset success!")

        ds_val = tfds.load(name="glue/sst2", split="validation", shuffle_files=False)
        SST2CharLevel.val_X, SST2CharLevel.val_y = prepare_ds(ds_val)
        print("Loading validation dataset success!")

        SST2CharLevel.test_X, SST2CharLevel.test_y = prepare_test_ds(open(a3t_sst2test_file).readlines())
        print("Loading test dataset success!")

    @staticmethod
    def make_vocab():
        assert SST2CharLevel.is_built
        return Vocab(SST2CharLevel)


class SST2WordLevel:
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    val_X = []
    val_y = []
    max_len = 56
    is_built = False
    vocab = None

    @staticmethod
    def build(dataset_home=DATASET_HOME):
        if SST2WordLevel.is_built:  # make sure it will only be built once
            return
        SST2WordLevel.is_built = True
        SST2WordLevel.vocab = Glove.make_vocab()

        sst2word_path = join(dataset_home, "sst2word")
        if not exists(sst2word_path):
            mkdir(sst2word_path)
        download_artifacts(dataset_home)
        a3t_sst2test_file = join(dataset_home, "A3T-artifacts", "dataset", "sst2test.txt")

        def prepare_ds(ds):
            X = []
            y = []
            for features in tfds.as_numpy(ds):
                sentence, label = features["sentence"], features["label"]
                tokens = word_tokenize(sentence.decode('UTF-8'))
                x = np.zeros(SST2WordLevel.max_len, dtype=np.int)
                for (i, token) in enumerate(tokens):
                    x[i] = SST2WordLevel.vocab.get_index(token)
                X.append(x)
                y.append(label)

            return np.array(X), np.array(y)

        def prepare_test_ds(ds):
            X = []
            y = []
            for features in ds:
                features = features.strip()
                sentence, label = features[2:], features[:1]
                tokens = word_tokenize(sentence)
                x = np.zeros(SST2WordLevel.max_len, dtype=np.int)
                for (i, token) in enumerate(tokens):
                    x[i] = SST2WordLevel.vocab.get_index(token)
                X.append(x)
                y.append(int(label))

            return np.array(X), np.array(y)

        ds_train = tfds.load(name="glue/sst2", split="train", shuffle_files=False)
        SST2WordLevel.train_X, SST2WordLevel.train_y = prepare_ds(ds_train)
        print("Loading training dataset success!")

        ds_val = tfds.load(name="glue/sst2", split="validation", shuffle_files=False)
        SST2WordLevel.val_X, SST2WordLevel.val_y = prepare_ds(ds_val)
        print("Loading validation dataset success!")

        SST2WordLevel.test_X, SST2WordLevel.test_y = prepare_test_ds(open(a3t_sst2test_file).readlines())
        print("Loading test dataset success!")
