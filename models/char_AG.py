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
        self.conv2d = Conv2D(64, 10, activation="relu")
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
        # layer = Gradient(loss)
        # partial = layer(look_up_c_d3)
        # self.partial_to_loss_model = Model(inputs=[self.c, self.y], outputs=partial)
        gradient = tf.gradients(loss, look_up_c_d3)[0]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        def lambda_partial(x, y):
            # return self.partial_to_loss_model.predict(x=[np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])[0]
            return \
            self.sess.run(gradient, feed_dict={self.c: np.expand_dims(x, axis=0), self.y: np.expand_dims(y, axis=0)})[0]

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
        # self.adv_model = Model(inputs=[self.c, self.adv], outputs=[self.adv_logits, self.logits])
        loss_layer = Lambda(lambda x: K.mean(categorical_crossentropy(self.y, x[0]) * 0.5 + categorical_crossentropy(self.y, x[1]) * 0.5))
        loss = loss_layer([self.adv_logits, self.logits])
        self.adv_model = Model(inputs=[self.c, self.adv, self.y], outputs=[loss])
        self.adv_model.add_loss(loss)
        def fn(y_true, y_pred):
            return categorical_crossentropy(y_true[0], y_pred[0]) * 0.5 + categorical_crossentropy(y_true[0], y_pred[1]) * 0.5        

        self.adv_model.compile(optimizer='RMSprop', loss=[None], loss_weights=[None])


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
    # model.model.load_weights("./tmp/%s" % saved_model_file)

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
                         SUB(lambda c: c in Alphabet.adjacent_keys, lambda c: Alphabet.adjacent_keys[c]),
                         keep_same)
    a = Composition(sub, sub, sub)
    # a = sub

    def adv_batch(batch_X, batch_Y):
        adv_batch_X = []
        arg_list = []
        for x, y in zip(batch_X, batch_Y):
            arg_list.append((chars.to_string(x), y, 1))
        rets = Multiprocessing.mapping(a.beam_search_adversarial, arg_list, 8, Alphabet.partial_to_loss)
        for ret in rets:
            # print(ret[0])
            adv_batch_X.append(chars.to_ids(ret[0]))
        return np.array(adv_batch_X)

    epochs = 30
    batch_size = 64
    pre_loss = 1e20
    patience = 5
    waiting = 0
    held_out = 1000
    for epoch in range(epochs):
        print("epoch %d:" % epoch)
        for i in range(0, training_num - held_out, batch_size):
            if i % 100 == 0: print(f'\radversarial training at %d/%d' % (i, training_num), flush=True)
            batch_X = training_X[i:min(training_num - held_out, i + batch_size)]
            batch_Y = training_Y[i:min(training_num - held_out, i + batch_size)]
            Alphabet.embedding = model.embed.get_weights()[0]
            adv_batch_X = adv_batch(batch_X, batch_Y)
            # print(model.model.evaluate(batch_X, batch_Y))
            # print(model.model.evaluate(adv_batch_X, batch_Y))
            # print(model.adv_model.evaluate(x=[batch_X, adv_batch_X, batch_Y], y=[]))
            model.adv_model.train_on_batch(x=[batch_X, adv_batch_X, batch_Y], y=[])

        Alphabet.embedding = model.embed.get_weights()[0]
        adv_batch_X = adv_batch(training_X[:held_out], training_Y[:held_out])
        loss = model.adv_model.evaluate(x=[training_X[:held_out], adv_batch_X, training_Y[:held_out]], y=[], batch_size=64)
        print("adv loss: %.2f" % loss)
        normal_loss, normal_acc = model.model.evaluate(x=test_X, y=test_Y, batch_size=64)
        print("normal loss: %.2f\t normal acc: %.2f" % (normal_loss, normal_acc))
        if loss > pre_loss:
            waiting += 1
            if waiting > patience:
                break
        else:
            waiting = 0
            pre_loss = loss

        model.adv_model.save_weights(filepath="./tmp/%s_epoch%d" % (adv_model_file, epochs))


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
    normal_loss, normal_acc = model.model.evaluate(test_X, test_Y, batch_size=64, verbose=0)
    print("normal loss: %.2f\t normal acc: %.2f" % (normal_loss, normal_acc))
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
                t = np.random.randint(0, len(s))
                while s[t] not in Alphabet.adjacent_keys:
                    t = np.random.randint(0, len(s))

                sub_chars = list(Alphabet.adjacent_keys[s[t]])
                id = np.random.randint(0, len(sub_chars))
                s = s[:t] + sub_chars[id] + s[t + 1:]

            X.append(chars.to_ids(s))
            Y.append(y)

        loss, acc = model.model.test_on_batch(np.array(X), np.array(Y))
        if acc == 1:
            adv_acc += 1

    print("adv acc: %.2f" % (adv_acc * 1.0 / len(test_Y)))
