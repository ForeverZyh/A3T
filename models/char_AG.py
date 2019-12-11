import keras
from keras.layers import Embedding, Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
import numpy as np


class char_AG:
    def __init__(self, all_voc_size=64, D=64):
        self.all_voc_size = all_voc_size
        self.D = D
        self.c = Input(shape=(300,), dtype='int32', name="input")
        self.embed = Embedding(self.all_voc_size, self.D, name="embedding")
        look_up_c = self.embed(self.c)
        look_up_c = Lambda(lambda x: K.expand_dims(x, -1))(look_up_c)
        conv2d = Conv2D(64, 10)
        x = conv2d(look_up_c)
        maxpooling = MaxPooling2D(10)
        x = maxpooling(x)
        x = Flatten()(x)
        fc1 = Dense(64)
        x = fc1(x)
        fc2 = Dense(64)
        x = fc2(x)
        fc3 = Dense(4, activation='softmax')
        self.logits = fc3(x)
        self.model = Model(inputs=self.c, outputs=self.logits)
        self.model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                                      verbose=0, mode='auto',
                                                                      baseline=None, restore_best_weights=False)


def train():
    training_X = np.load("../dataset/AG/X_train.npy")
    training_y = np.load("../dataset/AG/y_train.npy")
    test_X = np.load("../dataset/AG/X_test.npy")
    test_y = np.load("../dataset/AG/y_test.npy")
    nb_classes = 4
    training_Y = to_categorical(training_y, nb_classes)
    test_Y = to_categorical(test_y, nb_classes)

    model = char_AG()
    model.model.fit(x=training_X, y=training_Y, batch_size=64, epochs=30, callbacks=[model.early_stopping], verbose=2,
                    validation_data=(test_X[:500], test_Y[:500]), shuffle=True)
    model.model.save_weights(filepath="./tmp/char_AG")


train()
