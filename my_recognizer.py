import warnings
from asl_data import SinglesData
import operator
import sys


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for i in range(test_set.num_items):
        d = dict()
        for word in models.keys():
            X, length = test_set.get_item_Xlengths(i)
            logL = -sys.maxsize - 1
            try:
                logL = models[word].score(X, length)
            except:
                pass
            d[word] = logL

        probabilities.append(d)

        word = max(d, key=d.get)
        guesses.append(word)


    return probabilities, guesses
if __name__ == '__main__':
    from asl_data import AslDb
    import timeit

    asl= AslDb()

    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

    features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']

    training = asl.build_training(features_ground)
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()

    words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
    import timeit

    from my_model_selectors import SelectorCV
    from my_model_selectors import SelectorConstant


    def train_all_words(features, model_selector):
        training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()
        model_dict = {}
        for word in training.words:
            model = model_selector(sequences, Xlengths, word,
                                   n_constant=3).select()
            model_dict[word] = model
        return model_dict


    models = train_all_words(features_ground, SelectorCV)
    print("Number of word models returned = {}".format(len(models)))

    from asl_utils import show_errors

    test_set = asl.build_test(features_ground)
    probabilities, guesses = recognize(models, test_set)
    show_errors(guesses, test_set)
