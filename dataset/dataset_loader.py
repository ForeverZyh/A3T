import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize
import numpy as np


class Glove:
    str2id = {"_UNK_": 0}
    id2str = ["_UNK_"]
    embedding = []

    @staticmethod
    def get_word_id(s):
        if s in Glove.str2id:
            return Glove.str2id[s]
        # return the _UNK_
        return 0

    @staticmethod
    def build():
        glove_path = "./dataset/"
        try:
            Glove.embedding = np.load("./dataset/glove.npy")
            Glove.str2id = dict(np.load('./dataset/glove_dict.npy').item())
            Glove.id2str = [None] * len(Glove.str2id)
            for x in Glove.str2id:
                Glove.id2str[Glove.str2id[x]] = x
            print("Loading cached glove embedding success!")
            return
        except:
            pass

        glove_files = [f for f in listdir(glove_path) if isfile(join(glove_path, f)) and f[:5] == "glove"]
        if len(glove_files) == 0:
            raise AttributeError("No glove embeddings found in ./dataset/")
        else:
            glove_file = glove_files[0]
            print("Using ", glove_file, " ...")

        lines = open(join(glove_path, glove_file)).readlines()

        for (i, line) in enumerate(lines):
            tmp = line.strip().split()
            Glove.embedding.append(np.array([float(x) for x in tmp[1:]]))
            Glove.str2id[tmp[0]] = i + 1
            Glove.id2str.append(tmp[0])

        Glove.embedding = np.array(Glove.embedding)
        np.save("./dataset/glove", Glove.embedding)
        np.save("./dataset/glove_dict", Glove.str2id)
        print("Loading glove embedding success!")


class SSTWordLevel:
    training_X = []
    training_y = []
    test_X = []
    test_y = []
    val_X = []
    val_y = []
    synonym_dict = {}
    max_len = 60

    @staticmethod
    def synonym_dict_add_str(x, y):
        if x not in SSTWordLevel.synonym_dict:
            SSTWordLevel.synonym_dict[x] = set()

        SSTWordLevel.synonym_dict[x].add(y)

    @staticmethod
    def get_synonym():
        pddb_path = "./dataset/"
        try:
            SSTWordLevel.synonym_dict = dict(np.load("./dataset/synonym_dict.npy").item())
            print("Loading cached synonym_dict success!")
            return
        except:
            pass

        pddb_files = [f for f in listdir(pddb_path) if isfile(join(pddb_path, f)) and f[:4] == "ppdb"]
        if len(pddb_files) == 0:
            raise AttributeError("No PPDB files found in ./dataset/")
        else:
            pddb_file = pddb_files[0]
            print("Using ", pddb_files, " ...")

        lines = open(join(pddb_path, pddb_file)).readlines()
        for line in lines:
            tmp = line.strip().split(" ||| ")
            x, y = tmp[1], tmp[2]
            if x in Glove.str2id and y in Glove.str2id:
                SSTWordLevel.synonym_dict_add_str(x, y)
                SSTWordLevel.synonym_dict_add_str(y, x)

        np.save("./dataset/synonym_dict", SSTWordLevel.synonym_dict)
        print(SSTWordLevel.synonym_dict)
        print("Loading synonym_dict success!")

    @staticmethod
    def build():
        Glove.build()
        SSTWordLevel.get_synonym()

        tf.compat.v1.enable_eager_execution()

        def prepare_ds(ds, X, y):
            ds = ds.prefetch(10)
            for features in ds:
                sentence, label = features["sentence"], features["label"]
                tokens = word_tokenize(str(sentence.numpy()))
                x = np.zeros(SSTWordLevel.max_len)
                for (i, token) in enumerate(tokens):
                    x[i] = Glove.get_word_id(token)
                X.append(x)
                y.append(label.numpy())

            y = np.array(y)
            X = np.array(X)

        try:
            SSTWordLevel.training_X = np.load("./dataset/SST2/X_train.npy")
            SSTWordLevel.training_y = np.load("./dataset/SST2/y_train.npy")
            print("Loading cached training dataset success!")
        except:
            ds_train = tfds.load(name="glue/sst2", split="train", shuffle_files=False)
            prepare_ds(ds_train, SSTWordLevel.training_X, SSTWordLevel.training_y)
            np.save("./dataset/SST2/X_train", SSTWordLevel.training_X)
            np.save("./dataset/SST2/y_train", SSTWordLevel.training_y)
            print("Loading training dataset success!")

        try:
            SSTWordLevel.training_X = np.load("./dataset/SST2/X_val.npy")
            SSTWordLevel.training_y = np.load("./dataset/SST2/y_val.npy")
            print("Loading cached validation dataset success!")
        except:
            ds_val = tfds.load(name="glue/sst2", split="validation", shuffle_files=False)
            prepare_ds(ds_val, SSTWordLevel.val_X, SSTWordLevel.val_y)
            np.save("./dataset/SST2/X_val", SSTWordLevel.val_X)
            np.save("./dataset/SST2/y_val", SSTWordLevel.val_y)
            print("Loading validation dataset success!")

        try:
            SSTWordLevel.training_X = np.load("./dataset/SST2/X_test.npy")
            SSTWordLevel.training_y = np.load("./dataset/SST2/y_test.npy")
            print("Loading cached test dataset success!")
        except:
            ds_test = tfds.load(name="glue/sst2", split="test", shuffle_files=False)
            prepare_ds(ds_test, SSTWordLevel.test_X, SSTWordLevel.test_y)
            np.save("./dataset/SST2/X_test", SSTWordLevel.test_X)
            np.save("./dataset/SST2/y_test", SSTWordLevel.test_y)
            print("Loading test dataset success!")
