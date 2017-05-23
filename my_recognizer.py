import warnings
from asl_data import SinglesData
import operator
import sys
import arpa

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

def recognize_lm(probabilities: list, lm, test_set: SinglesData):
    """ Recognize test word sequences using obraines probabilities and language models

       :param probabilities: is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       :param lm : ARPA n-gram model
       :param test_set: SinglesData object
       :return: dictionary of recognized sentences: {video_num: sentence}
       """

    # Order the words by probabilities
    sorted_probs = list()
    for i in range(len(probabilities)):
        probs = [[probabilities[i][word], word] for word in probabilities[i]]
        sorted_probs.append(sorted(probs, key=lambda t: t[0], reverse=True))

    # Check only top TOP_NUM words
    TOP_NUM = min(50, len(probabilities[0]))
    ALPHA = 20.0
    recognized_sentences = dict()
    for video_num in test_set.sentences_index:
        sentence = list()
        for i in test_set.sentences_index[video_num]:
            max_score = float('-inf')
            recognized_word = ''
            for p_i in range(TOP_NUM):
                # Get the score from the trained model
                score = sorted_probs[i][p_i][0]

                try:
                    # Don't apply LM to the first word in the sentence
                    if len(sentence) > 0:
                        str_sentence = "<s> " + " ".join(sentence) + " " + sorted_probs[i][p_i][1]

                        # Get the log of p(current_word|previous_words)
                        lm_score = lm.log_p(str_sentence)

                        # Combine the scores using ALPHA model factor
                        score += ALPHA * lm_score
                except:
                    continue

                # Pick the word with the highest combined score
                if score > max_score:
                    max_score = score
                    recognized_word = sorted_probs[i][p_i][1]

            # Add the winning word to the sentence
            sentence.append(recognized_word)

        recognized_sentences[video_num] = sentence

    return recognized_sentences

def show_errors_lm(recognized_sentences: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """

    S = 0
    N = len(test_set.wordlist)

    print('Video  Recognized                                                    Correct')
    print('=====================================================================================================')
    for video_num in test_set.sentences_index:
        correct_sentence = [test_set.wordlist[i] for i in test_set.sentences_index[video_num]]
        recognized_sentence = recognized_sentences[video_num]
        for i in range(len(recognized_sentence)):
            if recognized_sentence[i] != correct_sentence[i]:
                recognized_sentence[i] = '*' + recognized_sentence[i]
                S += 1
        print('{:5}: {:60}  {}'.format(video_num, ' '.join(recognized_sentence), ' '.join(correct_sentence)))

    print("\n**** WER = {}".format(float(S) / float(N)))
    print("Total correct: {} out of {}".format(N - S, N))

if __name__ == '__main__':
    from asl_data import AslDb
    import pandas as pd
    import os
    import timeit

    models = arpa.loadf(os.path.join('data', 'ukn.3.lm'))
    lm = models[0]

    phrase = "<s> JOHN FISH WONT JOHN"
    # probability p(end|in, the)
    print(lm.log_p(phrase))

    # sentence score w/ sentence markers
    print(lm.log_s(phrase))

    phrase = "<s> JOHN FISH WONT EAT"
    # probability p(end|in, the)
    print(lm.log_p(phrase))

    # sentence score w/ sentence markers
    print(lm.log_s(phrase))

"""asl= AslDb()

    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

    features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']

    asl.df['delta-rx+4'] = asl.df['right-x'].diff(4).fillna(0)
    asl.df['delta-ry+4'] = asl.df['right-y'].diff(4).fillna(0)
    asl.df['delta-lx+4'] = asl.df['left-x'].diff(4).fillna(0)
    asl.df['delta-ly+4'] = asl.df['left-y'].diff(4).fillna(0)

    features_delta_4 = ['delta-rx+4', 'delta-ry+4', 'delta-lx+4', 'delta-ly+4']

    asl.df['delta-rx-4'] = asl.df['right-x'].diff(-4).fillna(0)
    asl.df['delta-ry-4'] = asl.df['right-y'].diff(-4).fillna(0)
    asl.df['delta-lx-4'] = asl.df['left-x'].diff(-4).fillna(0)
    asl.df['delta-ly-4'] = asl.df['left-y'].diff(-4).fillna(0)

    features_delta_4p = ['delta-rx-4', 'delta-ry-4', 'delta-lx-4', 'delta-ly-4']

    features_custom = features_ground + features_delta_4 + features_delta_4p

    df_min = asl.df.groupby('speaker').min()
    df_max = asl.df.groupby('speaker').max()


    def normalize(feature_name, feature_name_norm):
        feature_name_min = feature_name + "-min"
        feature_name_max = feature_name + "-max"

        asl.df[feature_name_min] = asl.df['speaker'].map(df_min[feature_name])
        asl.df[feature_name_max] = asl.df['speaker'].map(df_max[feature_name])

        asl.df[feature_name_norm] = (asl.df[feature_name] - asl.df[feature_name_min]) / (asl.df[feature_name_max] -
                                                                                         asl.df[feature_name_min])

    normalize('grnd-rx', 'grnd-rx-norm')
    normalize('grnd-lx', 'grnd-lx-norm')
    normalize('grnd-ry', 'grnd-ry-norm')
    normalize('grnd-ly', 'grnd-ly-norm')

    features_ground_norm = ['grnd-rx-norm', 'grnd-ry-norm', 'grnd-lx-norm', 'grnd-ly-norm']

    normalize('delta-rx+4', 'delta-rx+4-norm')
    normalize('delta-lx+4', 'delta-lx+4-norm')
    normalize('delta-ry+4', 'delta-ry+4-norm')
    normalize('delta-ly+4', 'delta-ly+4-norm')

    features_delta_4_norm = ['delta-rx+4-norm', 'delta-ry+4-norm', 'delta-lx+4-norm', 'delta-ly+4-norm']

    normalize('delta-rx-4', 'delta-rx-4-norm')
    normalize('delta-lx-4', 'delta-lx-4-norm')
    normalize('delta-ry-4', 'delta-ry-4-norm')
    normalize('delta-ly-4', 'delta-ly-4-norm')

    features_delta_4p_norm = ['delta-rx-4-norm', 'delta-ry-4-norm', 'delta-lx-4-norm', 'delta-ly-4-norm']

    features_custom_norm = features_ground_norm + features_delta_4_norm + features_delta_4p_norm

    from my_model_selectors import SelectorCV
    from my_model_selectors import SelectorBIC
    from my_model_selectors import SelectorDIC
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


    features = features_custom_norm

    models = train_all_words(features, SelectorBIC)
    print("Number of word models returned = {}".format(len(models)))

    from asl_utils import show_errors

    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    show_errors(guesses, test_set)

    df_probs = pd.DataFrame(data=probabilities)
    df_probs.head()
"""