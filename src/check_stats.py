import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def standardize_word(word):
    """Standardize word.

    A standard word is lower cased and only contains alphabetic
    characters. Return two strings: (std_word, illegal_chars)
    """
    # string containing the illegal characters found in the word
    illegal = ''
    # make word lower-case
    word = word.lower()
    if not word.isalpha():
        # there are non alphabetic chars in the word
        for char in word:
            # collect non alphabetic chars
            if not char.isalpha():
                if not char=='-':
                    illegal += char
        # remove non alphabetic chars from the word
        translation = str.maketrans({key: None for key in illegal})
        word = word.translate(translation) # -- Python 3
    return word#, illegal
 

def get_words(text):
    line_words = text.split()
    words = [standardize_word(w) for w in line_words]
    words = [w for w in words 
            if not w in set(stopwords.words('english'))
            and w!='']
    return words

def get_text_vectors(fname='../data/training_text'):

    all_words = []
    with open(fname, 'r') as f:
        for n, line in enumerate(f):
            if n:
                line_id, line_text = line.split('||')
                line_letters = re.sub("[^a-zA-Z,-]", " ", line_text) 
                line_letters_low = line_letters.lower()
                line_words = line_letters_low.split()
                line_words = [w for w in line_words
                        if not w in set(stopwords.words('english'))
                        and w!='']
                all_words.append(line_words)
                print(len(all_words), len(line_words))
    return all_words

    #train_text_df = pd.read_csv("../data/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])  

def get_basic_stats():
    genes_train_list = []
    variations_train_list = []
    genes_class_dic = {}
    variations_class_dic = {}
    training_variantsf = open('../data/training_variants')
    training_variants = training_variantsf.read()
    training_variantsf.close()

    training_variants = training_variants.split('\n')[:-1]
    for i in range(len(training_variants)-1):
        training_var = training_variants[i+1].split(',')
        gene = training_var[1]
        variation = training_var[2]
        cclass = training_var[3]    # cancer class

        if not gene in genes_train_list:
            genes_train_list.append(gene)
            genes_class_dic[gene] = [cclass]
        else:
            genes_class_dic[gene].append(cclass)

        if not variation in variations_train_list:
            variations_train_list.append(variation)
            variations_class_dic[variation] = [cclass]
        else:
            variations_class_dic[variation].append(cclass)

def get_cleaned_text(fname):
    all_words = []
    with open(fname, 'r') as f:
        for n, line in enumerate(f):
            if n:# and n<10:
                line_id, line_text = line.split('||')
                line_letters = re.sub("[^a-zA-Z -]", "", line_text) 
                line_letters_low = line_letters.lower()
                line_letters_low = stop_pattern.sub('', line_letters_low)
                line_words = line_letters_low.split()
                #all_words.append(line_words)
                all_words.append(line_letters_low)
                print(len(all_words), len(line_letters_low))
    return all_words


if __name__ == "__main__":

    training_variants = pd.read_csv('../data/training_variants')

    stop_pattern = re.compile(r'\b(' + r'|'.\
            join(stopwords.words('english')) + r')\b\s*')
    fname = '../data/training_text'
    train_text = get_cleaned_text(fname)

    #training_words = get_text_vectors(fname)
    import itertools
    flat_words = itertools.chain(*train_text)


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

    train_data_features = vectorizer.fit_transform(train_text)
    train_data_features = train_data_features.toarray()

    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit(train_data_features, training_variants["Class"])


    result_tr = forest.predict(train_data_features)
    print(np.corrcoef(result_tr, np.array(training_variants["Class"])))


    fname = '../data/test_text'
    test_text = get_cleaned_text(fname)
    test_data_features = vectorizer.transform(test_text)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)
    result_mat = np.zeros((len(result), 9))

    test_id = np.arange(len(result))

    result_mat[test_id, result-1] = 1
    result_mat = result_mat.astype(int)
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
    output.to_csv( "test.csv", index=False, quoting=3 )



    '''
    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_data_features, axis=0)

    for tag, count in zip(vocab, dist.tolist()[0]):
         print(tag, count)

    '''

