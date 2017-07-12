
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
                illegal += char
        # remove non alphabetic chars from the word
        translation = str.maketrans({key: None for key in illegal})
        word = word.translate(translation) # -- Python 3
    return word, illegal
 



if __name__ == "__main__":

    training_textf = open('../data/training_text')
    line_text = training_textf.readline()
    line_text = training_textf.readline()
    training_textf.close()


    line_id, line_text = line_text.split('||')


    genes_train_list = []
    variations_train_list = []
    training_variantsf = open('../data/training_variants')
    training_variants= training_variantsf.read()
    training_variantsf.close()

    training_variants = training_variants.split('\n')[:-1]
    for i in range(len(training_variants)-1):


        training_var = training_variants[i+1].split(',')
        if not training_var[1] in genes_train_list:
            genes_train_list.append(training_var[1])
        if not training_var[2] in variations_train_list:
            variations_train_list.append(training_var[2])




