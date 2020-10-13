import keras
from keras.layers import Embedding, Input, Dense, Lambda, Conv1D, Flatten, AveragePooling1D
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop
import numpy as np
import tensorflow as tf
import copy

from dataset.dataset_loader import SSTCharLevel
from DSL.transformations import REGEX, Transformation, INS, tUnion, SUB, DEL, Composition, Union, SWAP, TransformationDel, TransformationIns
from DSL.Alphabet import Alphabet
from utils import Dict, Multiprocessing, MultiprocessingWithoutPipe, Gradient, compute_adjacent_keys
import diffai.scheduling as S


SSTCharLevel.build() # this should be called first 
Alphabet.set_char_model()
Alphabet.max_len = SSTCharLevel.max_len
Alphabet.padding = " "
dict_map = SSTCharLevel.dict_map
Alphabet.set_alphabet(dict_map, np.zeros((len(dict_map), 150)))
S.Info.adjacent_keys = compute_adjacent_keys(dict_map)
keep_same = REGEX(r".*")
chars = Dict(dict_map)
sub = Transformation(keep_same, SUB(lambda c: c in Alphabet.adjacent_keys, lambda c: Alphabet.adjacent_keys[c]), keep_same)
swap = Transformation(keep_same, SWAP(lambda c: True, lambda c: True), keep_same)
ins = TransformationIns()
delete = TransformationDel()

class char_SST2:
    def __init__(self, lr=0.0009, all_voc_size=71, D=150):
        self.all_voc_size = all_voc_size
        self.D = D
        self.c = Input(shape=(Alphabet.max_len,), dtype='int32')
        self.y = Input(shape=(2,), dtype='float32')
        self.embed = Embedding(self.all_voc_size, self.D, name="embedding")
        look_up_c_d3 = self.embed(self.c)
        self.conv1d = Conv1D(100, 5, activation="relu")
        self.lr = lr
        x = self.conv1d(look_up_c_d3)
        self.avgpooling = AveragePooling1D(5)
        x = self.avgpooling(x)
        x = Flatten()(x)
        self.fc3 = Dense(2, activation='softmax')#, kernel_regularizer=keras.regularizers.l2(0.001))
        self.logits = self.fc3(x)
        self.model = Model(inputs=self.c, outputs=self.logits)
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5,
                                                                      verbose=0, mode='auto',
                                                                      baseline=None, restore_best_weights=True)
        loss = Lambda(lambda x: categorical_crossentropy(x[0], x[1]))([self.y, self.logits])
        layer = Gradient(loss)
        partial = layer(look_up_c_d3)
        self.partial_to_loss_model = Model(inputs=[self.c, self.y], outputs=partial)
        # gradient = tf.gradients(loss, look_up_c_d3)[0]
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

        def lambda_partial(x, y):
            return self.partial_to_loss_model.predict(x=[np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])[0]
#             return \
#                 self.sess.run(gradient,
#                               feed_dict={self.c: np.expand_dims(x, axis=0), self.y: np.expand_dims(y, axis=0)})[0]

        self.partial_to_loss = lambda_partial

    def adversarial_training(self):
        self.adv = Input(shape=(Alphabet.max_len,), dtype='int32')
        look_up_c = self.embed(self.adv)
        x = self.conv1d(look_up_c)
        x = self.avgpooling(x)
        x = Flatten()(x)
        self.adv_logits = self.fc3(x)
        loss_layer = Lambda(lambda x: K.mean(
            categorical_crossentropy(self.y, x[0]) * 0.5 + categorical_crossentropy(self.y, x[1]) * 0.5))
        loss = loss_layer([self.adv_logits, self.logits])
        self.adv_model = Model(inputs=[self.c, self.adv, self.y], outputs=[loss])
        self.adv_model.add_loss(loss)
        self.adv_model.compile(optimizer=Adam(learning_rate=self.lr), loss=[None], loss_weights=[None])


def train(filname, lr=0.0009):
    training_X = SSTCharLevel.training_X
    training_y = SSTCharLevel.training_y
    val_X = SSTCharLevel.val_X
    val_y = SSTCharLevel.val_y
    test_X = SSTCharLevel.test_X
    test_y = SSTCharLevel.test_y
    nb_classes = 2
    val_Y = to_categorical(val_y, nb_classes)
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)
    model = char_SST2(lr=lr)
    model.model.fit(x=training_X, y=training_Y, batch_size=64, epochs=30, callbacks=[model.early_stopping], verbose=2,
                    validation_data=(val_X, val_Y), shuffle=True)
    model.model.save_weights(filepath="./tmp/%s" % filname)
    normal_loss, normal_acc = model.model.evaluate(test_X, test_Y, batch_size=64, verbose=0)
    print("normal loss: %.4f\t normal acc: %.4f" % (normal_loss, normal_acc))
    return normal_loss, normal_acc


def adv_train(adv_model_file, target_transformation, adv_train_random=False, load_weights=None):
    training_X = SSTCharLevel.training_X
    training_y = SSTCharLevel.training_y
    val_X = SSTCharLevel.val_X
    val_y = SSTCharLevel.val_y
    test_X = SSTCharLevel.test_X
    test_y = SSTCharLevel.test_y
    nb_classes = 2
    val_Y = to_categorical(val_y, nb_classes)
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)
    training_num = len(training_X)
    
    model = char_SST2()
        
    model.adversarial_training()
    a = eval(target_transformation)
    if not adv_train_random:
        Alphabet.partial_to_loss = model.partial_to_loss

    def adv_batch(batch_X, batch_Y):
        adv_batch_X = []
        arg_list = []
        for x, y in zip(batch_X, batch_Y):
            arg_list.append((Alphabet.to_string(x, True), y, 1))
            # arg_list.append((chars.to_string(x), 1))
        rets = Multiprocessing.mapping(a.beam_search_adversarial, arg_list, 8, Alphabet.partial_to_loss)
        # rets = MultiprocessingWithoutPipe.mapping(a.random_sample, arg_list, 8)
        for i, ret in enumerate(rets):
            # print(ret)
            # print(chars.to_string(batch_X[i]))
            adv_batch_X.append(Alphabet.to_ids(ret[0][0]))
        return np.array(adv_batch_X)
    
    if load_weights is not None:
        model.model.load_weights("./tmp/%s" % load_weights)
        try:
            tmp = load_weights.split("_")[-1]
            if tmp[:5] == "epoch":
                starting_epoch = int(tmp[5:]) + 1
            else:
                starting_epoch = 0
        except:
            starting_epoch = 0
    else:
        starting_epoch = 0
        
    epochs = 20
    batch_size = 64
    pre_loss = 1e20
    patience = 5
    waiting = 0
    ids = np.arange(training_num)
    for epoch in range(starting_epoch, epochs):
        np.random.shuffle(ids)
        training_X = training_X[ids]
        training_Y = training_Y[ids]
        print("epoch %d:" % epoch)
        for i in range(0, training_num, batch_size):
            if i % 500 == 0: print('\radversarial training at %d/%d' % (i, training_num), flush=True)
            batch_X = training_X[i:min(training_num, i + batch_size)]
            batch_Y = training_Y[i:min(training_num, i + batch_size)]
            Alphabet.embedding = model.embed.get_weights()[0]
            adv_batch_X = adv_batch(batch_X, batch_Y)
            if i % 500 == 0:
                print(model.model.evaluate(batch_X, batch_Y))
                print(model.model.evaluate(adv_batch_X, batch_Y))
                # print(model.adv_model.evaluate(x=[batch_X, adv_batch_X, batch_Y], y=[]))
            model.adv_model.train_on_batch(x=[batch_X, adv_batch_X, batch_Y], y=[])

        Alphabet.embedding = model.embed.get_weights()[0]
        adv_batch_X = adv_batch(val_X, val_Y)
        #loss = model.adv_model.evaluate(x=[val_X, adv_batch_X, val_Y], y=[],
        #                                batch_size=64)
        _, acc0 = model.model.evaluate(x=val_X, y=val_Y, batch_size=64)
        _, acc1 = model.model.evaluate(x=adv_batch_X, y=val_Y, batch_size=64)
        loss = - acc0 - acc1
        print("adv loss: %.4f" % loss)
        normal_loss, normal_acc = model.model.evaluate(x=test_X, y=test_Y, batch_size=64)
        print("normal loss: %.4f\t normal acc: %.4f" % (normal_loss, normal_acc))
        if loss > pre_loss:
            waiting += 1
            if waiting >= patience:
                break
        else:
            waiting = 0
            pre_loss = loss
            model.adv_model.save_weights(filepath="./tmp/%s" % adv_model_file)

#         model.adv_model.save_weights(filepath="./tmp/%s_epoch%d" % (adv_model_file, epoch))

   # model.adv_model.save_weights(filepath="./tmp/%s" % adv_model_file)


def test_model(saved_model_file, func=None, target_transformation="None", test_only=False):
    training_X = SSTCharLevel.training_X
    training_y = SSTCharLevel.training_y
    test_X = SSTCharLevel.test_X
    test_y = SSTCharLevel.test_y
    nb_classes = 2
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)
    training_num = len(training_X)
    testing_num = len(test_X)

    model = char_SST2()
    model.model.load_weights("./tmp/%s" % saved_model_file)
    normal_loss, normal_acc = model.model.evaluate(test_X, test_Y, batch_size=64, verbose=0)
    print("normal loss: %.4f\t normal acc: %.4f" % (normal_loss, normal_acc))
    if test_only:
        return
    model.adversarial_training()
    a = eval(target_transformation)
    Alphabet.partial_to_loss = model.partial_to_loss
    
    def adv_batch(batch_X, batch_Y):
        adv_batch_X = []
        arg_list = []
        for x, y in zip(batch_X, batch_Y):
            arg_list.append((Alphabet.to_string(x, True), y, 1))
        rets = Multiprocessing.mapping(a.beam_search_adversarial, arg_list, 8, Alphabet.partial_to_loss)
        for i, ret in enumerate(rets):
            adv_batch_X.append(Alphabet.to_ids(ret[0][0]))
        return np.array(adv_batch_X)
    
    correct = 0
    batch_size = 64
    if func is not None:
        for i, (x, y) in enumerate(zip(test_X, test_Y)):
            oracle_iterator = func(x, True, batch_size)
            all_correct = True
            for batch_X in oracle_iterator:
                batch_Y = np.tile(np.expand_dims(y, 0), (len(batch_X), 1))
                loss, acc = model.model.test_on_batch(batch_X, batch_Y)
                if acc != 1:
                    all_correct = False
                    break
                
            if all_correct: correct += 1
            if (i + 1) % 100 == 0:
                print(i + 1, correct * 100.0 / (i + 1))
        
        print("oracle acc: %.4f" % (correct * 100.0 / len(test_Y)))
    else:
        adv_acc = 0 
        Alphabet.embedding = model.embed.get_weights()[0]
        for i in range(0, testing_num, batch_size):
            if i % 100 == 0: print('\radversarial testing at %d/%d' % (i, testing_num), flush=True)
            batch_X = test_X[i:min(training_num, i + batch_size)]
            batch_Y = test_Y[i:min(training_num, i + batch_size)]
            adv_batch_X = adv_batch(batch_X, batch_Y)
            loss, acc = model.model.evaluate(adv_batch_X, batch_Y)
            if i % 100 == 0: print(loss, acc)
            adv_acc += acc * len(batch_X)

        print("adv acc: %.4f" % (adv_acc * 1.0 / len(test_Y)))
