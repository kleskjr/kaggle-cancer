import pandas as pd
import numpy as np
import re
import importlib

from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Masking
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Dropout, Input, LSTM, MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from gensim.models import Word2Vec, KeyedVectors 

from keras.utils.np_utils import to_categorical

from keras import backend as K
import tensorflow as tf

import cnn

from stopwords_list import *
stop_pattern = re.compile(r'(?<!-)\b(' \
                        + r'|'.join(stopwords.words('english')+stop_words) \
                        + r')\b(?!-)\s*')

n_classes = 9

def get_cleaned_text(fname, max_len=999999):
    all_words = []
    with open(fname, 'r') as f:
        for n, line in enumerate(f):
            if n and n<max_len:
                line_id, line_text = line.split('||')
                # this regex should remove all the lone numbers,
                # i.e, numbers that are not part of words
                #line_text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", line_text) 
                line_text = re.sub(r"(?<!-)\b\d+\b", " ", line_text)
                # keep letters, numbers and a few characters that are 
                # seen in words; substitute other characters with a space
                line_text = re.sub("[^\w\-\*\_]", " ", line_text)

                line_letters_low = line_text.lower()
                line_letters_low = stop_pattern.sub(' ', line_letters_low)
                line_words = line_letters_low.split()
                line_words = [s for s in line_words if len(s) > 2]
                all_words.append(line_words)
    return all_words


def get_feature_vector2(fname_text, variants, n_text=999999):
    '''
    from each text get a list (an array) of w2v
    version 1 was taking the mean w2v

    '''
    text = get_cleaned_text(fname_text, n_text)

    text_wv = []
    # creating feature vector from text by stacking w2vs
    for i, text in enumerate(text):
        text_wv.append(np.array([w2v_model.wv[w] for w in text 
                        if w in w2v_model.wv.vocab]))

    # set same length for all texts; appending zeros for smaller texts 
    maxlen = max([len(t) for t in text_wv])
    text_wv_padded = \
        [np.pad(t, ((0, maxlen-len(t)), (0, 0)),
            'constant', constant_values=0) for t in text_wv]

    return np.array(text_wv_padded)


def create_w2v(vector_size=200):
    '''
        Creates w2v model based on train and test texts
        The trained model looks good enough, if not better than the
        pretrained PubMed models in the net

    '''
    text_train = get_cleaned_text('../data/test_text')
    text_test = get_cleaned_text('../data/training_text')
    w2v_model = Word2Vec(text_train+text_test, size=vector_size, 
                        window=5, min_count=3, workers=4, iter=5)
    return w2v_model 


if __name__ == '__main__':
    train_variants = pd.read_csv('../data/training_variants')
    test_variants = pd.read_csv('../data/test_variants')

    n_text = len(train_variants)
    #w2v_model  = create_w2v()
    #w2v_model = Word2Vec.load('w2v_tt200')
    w2v_model = Word2Vec.load('w2v_tt50')

    fvec = get_feature_vector2('../data/training_text',
                                train_variants, n_text+1)

    wvec_size = w2v_model.layer1_size

    #nn_model = cnn.test(fvec2)
    nn_model = cnn.create_model(fvec[:,:,:], 
                                train_variants, epochs=10)

    fvec_test = get_feature_vector2('../data/test_text', test_variants)
    cnn.predict_cnn(nn_model, fvec_test)


