import keras
from keras.layers import Embedding, Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
import numpy as np
import tensorflow as tf

from dataset.dataset_loader import SSTWordLevel, Glove


class word_SST2:
    def __init__(self):
        SSTWordLevel.build()
        self.c = Input(shape=(SSTWordLevel.max_len,), dtype='int32')
        self.y = Input(shape=(2,), dtype='float32')
        self.all_voc_size, self.D = Glove.embedding.shape
        self.embed = Embedding(self.all_voc_size, self.D, name="embedding", trainable=False, weights=[Glove.embedding])
        look_up_c_d3 = self.embed(self.c)
        look_up_c = Lambda(lambda x: K.expand_dims(x, -1))(look_up_c_d3)
        self.conv2d = Conv2D(100, 5, activation="relu")
        x = self.conv2d(look_up_c)
        self.maxpooling = MaxPooling2D(5)
        x = self.maxpooling(x)
        x = Flatten()(x)
        self.fc = Dense(2, activation='softmax')
        self.logits = self.fc(x)
        self.model = Model(inputs=self.c, outputs=self.logits)
        self.model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                                      verbose=0, mode='auto',
                                                                      baseline=None, restore_best_weights=False)
        loss = Lambda(lambda x: categorical_crossentropy(x[0], x[1]))([self.y, self.logits])
        gradient = tf.gradients(loss, look_up_c_d3)[0]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        def lambda_partial(x, y):
            return \
                self.sess.run(gradient,
                              feed_dict={self.c: np.expand_dims(x, axis=0), self.y: np.expand_dims(y, axis=0)})[0]

        self.partial_to_loss = lambda_partial

    def adversarial_training(self):
        self.adv = Input(shape=(SSTWordLevel.max_len,), dtype='int32')
        look_up_c = self.embed(self.adv)
        look_up_c = Lambda(lambda x: K.expand_dims(x, -1))(look_up_c)
        x = self.conv2d(look_up_c)
        x = self.maxpooling(x)
        x = Flatten()(x)
        self.adv_logits = self.fc(x)
        loss_layer = Lambda(lambda x: K.mean(
            categorical_crossentropy(self.y, x[0]) * 0.5 + categorical_crossentropy(self.y, x[1]) * 0.5))
        loss = loss_layer([self.adv_logits, self.logits])
        self.adv_model = Model(inputs=[self.c, self.adv, self.y], outputs=[loss])
        self.adv_model.add_loss(loss)
        self.adv_model.compile(optimizer='RMSprop', loss=[None], loss_weights=[None])


def train(filname):
    model = word_SST2()  # this should be called first
    training_X = SSTWordLevel.training_X
    training_y = SSTWordLevel.training_y
    val_X = SSTWordLevel.val_X
    val_y = SSTWordLevel.val_y
    nb_classes = 4
    training_Y = to_categorical(training_y, nb_classes)
    val_Y = to_categorical(val_y, nb_classes)

    model.model.fit(x=training_X, y=training_Y, batch_size=64, epochs=30, callbacks=[model.early_stopping], verbose=2,
                    validation_data=(val_X, val_Y), shuffle=True)
    model.model.save_weights(filepath="./tmp/%s" % filname)


def test_model(saved_model_file):
    model = word_SST2()
    test_X = SSTWordLevel.test_X
    test_y = SSTWordLevel.test_y
    nb_classes = 4
    test_Y = to_categorical(test_y, nb_classes)

    model.model.load_weights("./tmp/%s" % saved_model_file)
    normal_loss, normal_acc = model.model.evaluate(test_X, test_Y, batch_size=64, verbose=0)
    print("normal loss: %.2f\t normal acc: %.2f" % (normal_loss, normal_acc))
