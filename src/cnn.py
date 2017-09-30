
from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Masking, Flatten
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Dropout, Input, LSTM, MaxPooling1D, Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D

from keras.utils.np_utils import to_categorical

from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

n_classes = 9


def create_model(fvec, train_variants, epochs=2):
    
    wvec_size = fvec.shape[2]
    main_input = Input(shape=(None, wvec_size), name='main_input')
    x10 = Conv1D(5, 10, padding='same',
                input_shape=(None, wvec_size))(main_input)
    x10 = Activation('relu')(x10)
    x10 = Dropout(0.2)(x10)

    x5 = Conv1D(8, 5, padding='same',
                input_shape=(None, wvec_size))(main_input)
    x5 = Activation('relu')(x5)
    x5 = Dropout(0.2)(x5)

    x3 = Conv1D(10, 3, padding='same',
                input_shape=(None, wvec_size))(main_input)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.2)(x3)

    #x3 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(x3)

    x = Concatenate()([x10, x5, x3])
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    z = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat',
            weights=None)(x)

    #x = Conv1D(20, 10, padding='same')(z)
    #x = Activation('relu')(x)
    #x = Dropout(0.1)(x)

    #x = Conv1D(10, 5, padding='same')(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.1)(x)
    #x = GlobalMaxPooling1D()(x)

    #x = Activation('tanh')(x)
    #x = Dense(20)(x)
    #x = Activation('tanh')(x)


    x = Conv1D(10, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling1D()(x)

    #x = TimeDistributed(Dense(10))(10)
    #x = Activation('tanh')(x)

    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    nn_model = Model(inputs=[main_input], outputs=output)
    nn_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    y_binary = to_categorical(np.array(
                    train_variants["Class"].values)-1, n_classes)
    nn_model.fit(fvec, y_binary, epochs=epochs, validation_split=0.2)
 
    return nn_model


def create_model_min(fvec, train_variants, epochs=2):
    
    wvec_size = fvec.shape[2]
    text_len = None#fvec.shape[1]#None
    main_input = Input(shape=(text_len, wvec_size))
    x = Conv1D(10, 10, padding='valid', input_shape=(text_len, wvec_size), 
            kernel_initializer='ones')(main_input)

    '''
    #x = MaxPooling1D(pool_size=100, strides=None, padding='valid')(x)
    #x = Flatten()(x)
    x = GlobalMaxPooling1D()(x)
    # without an activation after globalmaxpooling, there is no output
    x = Activation('tanh')(x)
    '''
    z = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat',
            weights=None)(x)

    x = Conv1D(20, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling1D()(x)

    #1/0
    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    nn_model = Model(inputs=main_input, outputs=output)
    nn_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    y_binary = to_categorical(np.array(
                    train_variants["Class"].values)-1, n_classes)
    nn_model.fit(fvec, y_binary, epochs=epochs, validation_split=0.1)
 
    return nn_model


def create_model_kur(fvec, a, epochs):

    #Embedding(num_words, embed_dim, input_length = X.shape[1])

    wvec_size = fvec.shape[2]
    text_len = None#fvec.shape[1]#None
    nn_model = Sequential()
    nn_model.add(Embedding(fvec.shape[1], fvec.shape[2], fvec.shape[0]))
    #nn_model.add(Input(shape=(text_len, wvec_size)))
    nn_model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    nn_model.add(Dense(9,activation='softmax'))
    nn_model.compile(loss = 'categorical_crossentropy', optimizer='adam',
                    metrics = ['categorical_crossentropy'])
    print(nn_model.summary())
    return nn_model


def test(fvec):

    #nn_model = create_model_n(fvec, epochs=100)
    #predict_cnn(nn_model, fvec[:10,:,:])

    n_texts = 10
    train_variants = pd.read_csv('../data/training_variants')[:n_texts]
    #test_variants = pd.read_csv('../data/test_variants')

    nn_model = create_model_min(fvec[:n_texts,:2000,:2], 
                                train_variants, epochs=100)
    predict_cnn(nn_model, fvec[:n_texts,:2000,:2])

    return nn_model



