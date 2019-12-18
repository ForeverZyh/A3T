import keras
from keras.layers import Embedding, Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
import numpy as np
import tensorflow as tf

from DSL.transformations import REGEX, Transformation, INS, tUnion, SUB, DEL, Composition, Union
from DSL.Alphabet import Alphabet
from utils import Dict, Gradient, Multiprocessing


class char_AG:
    def __init__(self, all_voc_size=64, D=64):
        self.all_voc_size = all_voc_size
        self.D = D
        self.c = Input(shape=(300,), dtype='int32')
        self.y = Input(shape=(4,), dtype='float32')
        self.embed = Embedding(self.all_voc_size, self.D, name="embedding")
        look_up_c_d3 = self.embed(self.c)
        look_up_c = Lambda(lambda x: K.expand_dims(x, -1))(look_up_c_d3)
        self.conv2d = Conv2D(64, 10)
        x = self.conv2d(look_up_c)
        self.maxpooling = MaxPooling2D(10)
        x = self.maxpooling(x)
        x = Flatten()(x)
        self.fc1 = Dense(64)
        x = self.fc1(x)
        self.fc2 = Dense(64)
        x = self.fc2(x)
        self.fc3 = Dense(4, activation='softmax')
        self.logits = self.fc3(x)
        self.model = Model(inputs=self.c, outputs=self.logits)
        self.model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                                      verbose=0, mode='auto',
                                                                      baseline=None, restore_best_weights=False)
        loss = Lambda(lambda x: categorical_crossentropy(x[0], x[1]))([self.y, self.logits])
        layer = Gradient(loss)
        partial = layer(look_up_c_d3)
        self.partial_to_loss_model = Model(inputs=[self.c, self.y], outputs=partial)

        def lambda_partial(x, y):
            return self.partial_to_loss_model.predict(x=[np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])[0]

        self.partial_to_loss = lambda_partial

    def adversarial_training(self):
        self.adv = Input(shape=(300,), dtype='int32')
        look_up_c = self.embed(self.adv)
        look_up_c = Lambda(lambda x: K.expand_dims(x, -1))(look_up_c)
        x = self.conv2d(look_up_c)
        x = self.maxpooling(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        self.adv_logits = self.fc3(x)
        self.weighted_logits = Lambda(lambda x: x[0] * 0.5 + x[1] * 0.5)([self.adv_logits, self.logits])
        self.adv_model = Model(inputs=[self.c, self.adv], outputs=self.weighted_logits)
        self.adv_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])


def train(filname):
    training_X = np.load("./dataset/AG/X_train.npy")
    training_y = np.load("./dataset/AG/y_train.npy")
    test_X = np.load("./dataset/AG/X_test.npy")
    test_y = np.load("./dataset/AG/y_test.npy")
    nb_classes = 4
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)

    model = char_AG()
    model.model.fit(x=training_X, y=training_Y, batch_size=64, epochs=30, callbacks=[model.early_stopping], verbose=2,
                    validation_data=(training_X[:500], training_Y[:500]), shuffle=True)
    model.model.save_weights(filepath="./tmp/%s" % filname)


def adv_train(saved_model_file, adv_model_file):
    training_X = np.load("./dataset/AG/X_train.npy")
    training_y = np.load("./dataset/AG/y_train.npy")
    test_X = np.load("./dataset/AG/X_test.npy")
    test_y = np.load("./dataset/AG/y_test.npy")
    nb_classes = 4
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)
    training_num = len(training_X)

    model = char_AG()
    model.model.load_weights("./tmp/%s" % saved_model_file)

    model.adversarial_training()
    Alphabet.set_char_model()
    Alphabet.partial_to_loss = model.partial_to_loss
    Alphabet.max_len = 300
    Alphabet.padding = " "
    dict_map = dict(np.load("./dataset/AG/dict_map.npy").item())
    Alphabet.set_alphabet(dict_map, np.zeros((56, 64)))
    keep_same = REGEX(r".*")
    chars = Dict(dict_map)
    sub = Transformation(keep_same,
                         SUB(lambda c: c in Alphabet.adjacent_keys, lambda c: set(Alphabet.adjacent_keys[c]).intersection(set(Alphabet.alphabet))),
                         keep_same)
    # a = Composition(sub, sub)
    a = sub

    def adv_batch(batch_X, batch_Y):
        adv_batch_X = []
        arg_list = []
        for x, y in zip(batch_X, batch_Y):
            arg_list.append((chars.to_string(x), y, 3))

        # rets = Multiprocessing.mapping(a.beam_search_adversarial, arg_list, 1)
        # print(rets)
        for arg in arg_list:
            ret = a.beam_search_adversarial(*arg)
            adv_batch_X.append(chars.to_ids(ret[0][0]))

        return np.array(adv_batch_X)

    epochs = 30
    batch_size = 64
    pre_loss = 1e20
    patience = 5
    waiting = 0
    for epoch in range(epochs):
        print("epoch %d:" % epoch)
        for i in range(0, training_num, batch_size):
            print(f'\radversarial training at %d/%d' % (i, training_num), flush=True)
            batch_X = training_X[i:min(training_num, i + batch_size)]
            batch_Y = training_Y[i:min(training_num, i + batch_size)]
            Alphabet.embedding = model.embed.get_weights()[0]
            adv_batch_X = adv_batch(batch_X, batch_Y)
            model.adv_model.train_on_batch(x=[batch_X, adv_batch_X], y=batch_Y)

        Alphabet.embedding = model.embed.get_weights()[0]
        adv_batch_X = adv_batch(training_X[:500], training_Y[:500])
        loss, acc = model.adv_model.evaluate(x=[training_X[:500], adv_batch_X], y=training_Y[:500], batch_size=64)
        print("adv loss: %.2f\t adv acc: %.2f" % (loss, acc))
        if loss > pre_loss:
            waiting += 1
            if waiting > patience:
                break
        else:
            waiting = 0
            pre_loss = loss

    model.adv_model.save_weights(filepath="./tmp/%s" % adv_model_file)


def test_model(saved_model_file):
    training_X = np.load("./dataset/AG/X_train.npy")
    training_y = np.load("./dataset/AG/y_train.npy")
    test_X = np.load("./dataset/AG/X_test.npy")
    test_y = np.load("./dataset/AG/y_test.npy")
    nb_classes = 4
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)
    training_num = len(training_X)

    model = char_AG()
    model.model.load_weights("./tmp/%s" % saved_model_file)
    # normal_loss, normal_acc = model.model.evaluate(test_X, test_Y, batch_size=64, verbose=0)
    # print("normal loss: %.2f\t normal acc: %.2f" % (normal_loss, normal_acc))
    dict_map = dict(np.load("./dataset/AG/dict_map.npy").item())
    Alphabet.set_char_model()
    Alphabet.set_alphabet(dict_map, np.zeros((56, 64)))
    chars = Dict(dict_map)
    adv_acc = 0
    for x, y in zip(test_X, test_Y):
        X = []
        Y = []
        for _ in range(64):
            s = chars.to_string(x)
            for subs in range(1):
                sub_chars = []
                while True:
                    t = np.random.randint(0, len(s))
                    if s[t] not in Alphabet.adjacent_keys:
                        continue
                    sub_chars = list(set(Alphabet.adjacent_keys[s[t]]).intersection(Alphabet.alphabet))
                    if len(sub_chars) == 0:
                        continue
                    id = np.random.randint(0, len(sub_chars))
                    break

                s = s[:t] + sub_chars[id] + s[t + 1:]

            X.append(chars.to_ids(s))
            Y.append(y)

        loss, acc = model.model.test_on_batch(np.array(X), np.array(Y))
        if acc == 1:
            adv_acc += 1

    print("adv acc: %.2f" % (adv_acc * 1.0 / len(test_Y)))
