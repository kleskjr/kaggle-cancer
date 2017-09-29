
from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Masking, Flatten
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Dropout, Input, LSTM, MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D

from keras.utils.np_utils import to_categorical

from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

n_classes = 9


def create_model(fvec, epochs=2):
    
    wvec_size = fvec.shape[2]
    main_input = Input(shape=(None, wvec_size), name='main_input')
    x10 = Conv1D(5, 10, padding='same',
                input_shape=(None, wvec_size))(main_input)
    #x10 = Activation('relu')(x10)
    #x10 = Dropout(0.2)(x10)

    x5 = Conv1D(8, 5, padding='same',
                input_shape=(None, wvec_size))(main_input)
    #x5 = Activation('relu')(x5)
    #x5 = Dropout(0.2)(x5)

    x3 = Conv1D(10, 3, padding='same',
                input_shape=(None, wvec_size))(main_input)
    #x3 = Activation('relu')(x3)
    #x3 = Dropout(0.2)(x3)

    #x3 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(x3)

    x = Concatenate()([x10, x5, x3])
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    x = Conv1D(20, 10, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv1D(10, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = GlobalMaxPooling1D()(x)

    x = Activation('tanh')(x)
    x = Dense(20)(x)
    x = Activation('tanh')(x)
    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    nn_model = Model(inputs=[main_input], outputs=output)
    nn_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    y_binary = to_categorical(np.array(
                    train_variants["Class"].values)-1, n_classes)
    nn_model.fit(fvec, y_binary, epochs=epochs, validation_split=0.2)
 
    return nn_model


def create_model_n(fvec, train_variants, epochs=2):
    
    wvec_size = fvec.shape[2]
    main_input = Input(shape=(None, wvec_size), name='main_input')

    x = Conv1D(1, 1, padding='valid',
                input_shape=(None, wvec_size))(main_input)
    x = Activation('tanh')(x)

    #x = MaxPooling1D(pool_size=5, strides=None, padding='valid')(x)
    #x = Flatten(input_shape=(None, 1))(x)

    x = GlobalMaxPooling1D()(x)
    x = Activation('tanh')(x)

    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    nn_model = Model(inputs=[main_input], outputs=output)
    nn_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    y_binary = to_categorical(np.array(
                    train_variants["Class"].values)-1, n_classes)
    nn_model.fit(fvec, y_binary, epochs=epochs, validation_split=0.2)
 
    return nn_model


def predict_cnn(nn_model, fvec_test):
    
    #test_variants = pd.read_csv('../data/test_variants')
    #fvec_test = get_feature_vector2('../data/test_text', train_variants)

    result_mat = nn_model.predict(fvec_test)
    test_id = np.arange(len(result_mat))

    #result_mat = result_mat.astype(int)
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

    #nn_model = create_model_n(fvec, epochs=100)
    #predict_cnn(nn_model, fvec[:10,:,:])

    n_texts = 10
    train_variants = pd.read_csv('../data/training_variants')[:n_texts]
    test_variants = pd.read_csv('../data/test_variants')

    nn_model = create_model_n(fvec[:n_texts,:,:2], train_variants, epochs=10)
    predict_cnn(nn_model, fvec[:n_texts,:,:2])



