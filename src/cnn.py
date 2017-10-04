
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
    '''
    Creates and traines a minimal CNN with a LSTM module and global
    pooling of extracted features in the temporal dimension.

    '''
    wvec_size = fvec.shape[2]
    text_len = None#fvec.shape[1]
    main_input = Input(shape=(text_len, wvec_size))

    x10 = Conv1D(10, 10, padding='same', input_shape=(text_len, wvec_size), 
            kernel_initializer='ones')(main_input)
    x10 = Activation('relu')(x10)
    x10 = Dropout(0.2)(x10)

    x5 = Conv1D(20, 5, padding='same', input_shape=(text_len, wvec_size), 
            kernel_initializer='ones')(main_input)
    x5 = Activation('relu')(x5)
    x5 = Dropout(0.2)(x5)

    x3 = Conv1D(30, 3, padding='same', input_shape=(text_len, wvec_size), 
            kernel_initializer='ones')(main_input)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.2)(x3)

    x = Concatenate()([x10, x5, x3])

    '''
    #x = MaxPooling1D(pool_size=100, strides=None, padding='valid')(x)
    #x = Flatten()(x)
    x = GlobalMaxPooling1D()(x)
    # without an activation after globalmaxpooling, there is no output
    x = Activation('tanh')(x)
    '''
    z = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat',
            weights=None)(x)

    x10 = Conv1D(10, 10, padding='same')(z)
    x10 = Activation('relu')(x10)
    x10 = Dropout(0.2)(x10)

    x5 = Conv1D(20, 5, padding='same')(z)
    x5 = Activation('relu')(x5)
    x5 = Dropout(0.2)(x5)

    x = Concatenate()([x10, x5])

    x = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D(pool_size=100, strides=50, padding='valid')(x)
    #x = Flatten()(x)

    #x = Dense(80)(x)
    #output = Activation('tanh')(x)

    x = Dense(30)(x)
    output = Activation('tanh')(x)

    x = Dense(10)(x)
    output = Activation('tanh')(x)

    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    nn_model = Model(inputs=main_input, outputs=output)
    nn_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    y_binary = to_categorical(np.array(
                    train_variants["Class"].values)-1, n_classes)
    nn_model.fit(fvec, y_binary, epochs=epochs, validation_split=0.2)
 
    return nn_model


def predict_cnn(nn_model, fvec_test):
    
    result_mat = nn_model.predict(fvec_test)
    test_id = np.arange(len(result_mat))

    output = pd.DataFrame( data={"ID":test_id, "class1":result_mat[:,0],
            "class2":result_mat[:,1],
            "class3":result_mat[:,2],
            "class4":result_mat[:,3],
            "class5":result_mat[:,4],
            "class6":result_mat[:,5],
            "class7":result_mat[:,6],
            "class8":result_mat[:,7],
            "class9":result_mat[:,8]
            } )
    output.to_csv( "compute2_test.csv", index=False)


def test(fvec):

    train_variants = pd.read_csv('../data/training_variants')[:]

    nn_model = create_model(fvec[:,:,:], 
                            train_variants, epochs=10)
    predict_cnn(nn_model, fvec)

    return nn_model



