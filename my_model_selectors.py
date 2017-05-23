import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import sys


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=2,
                 min_n_components=2, max_n_components=30,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # The list of BIC scores for the models
        lBICs = list()
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            try:
                if model:
                    logL = model.score(self.X, self.lengths)

                    # Calculate the number of parameter, p, in the BIC criteria
                    p = n_components * n_components + 2 * n_components * len(self.X[0]) - 1
                    BIC = -2.0 * logL + p * np.log(len(self.X))
                    lBICs.append([n_components, BIC, model])
            except:
                # The score method throws an exception, ignore the current model
                pass

        if self.verbose:
            print("BIC values [states, BIC]: {}".format(lBICs))

        # If the list is empty, GaussianHMM could not train any models for the given word
        if len(lBICs) == 0:
            return None

        # Get the model with the minimum BIC score
        n_components, BIC, model = min(lBICs, key=lambda e: e[1])

        if self.verbose:
            print("Min BIC is {} when {} states".format(BIC, n_components))
            print("Best model: {}", model)

        return model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        lDICs = list()
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            if model:
                try:
                    # Get the average likelihood value for self.this_word
                    logL = model.score(self.X, self.lengths) / len(self.lengths)
                except:
                    # Can't get the score - skip this model
                    continue

                sum_logL = 0.0
                count = 0
                for word in self.hwords.keys():
                    if word is not self.this_word:
                        try:
                            other_X, other_lengths = self.hwords[word]
                            # Get the average anti-likelihood value for word
                            sum_logL += model.score(other_X, other_lengths) / len(other_lengths)
                            count += 1
                        except:
                            # Can't get the score - skip this model
                            pass

                DIC = logL
                if self.verbose:
                    print("likelihood: {}".format(logL))
                if count > 0:
                    # Get the DIC score
                    DIC -= sum_logL / count
                    if self.verbose:
                        print("anti likelihood: {}".format(sum_logL / count))

                lDICs.append([n_components, DIC, model])

        # If the list is empty, GaussianHMM could not train any models for the given word
        if len(lDICs) == 0:
            return None

        # Get the model with the maximum DIC score
        n_components, DIC, model = max(lDICs, key=lambda e: e[1])

        return model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.lengths) == 1:
            return self.base_model(self.n_constant)

        # Create 3-fold split method (if possible)
        split_method = KFold(min(3, len(self.lengths)))
        best_logL = float('-inf')
        best_n_components = 0
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            total_logL = 0
            count = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # Use the training Xlength values
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                model = self.base_model(n_components)

                try:
                    if model:
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                        # Get the likelihood score
                        total_logL += model.score(test_X, test_lengths)
                        count += 1
                except:
                    if self.verbose:
                        print("failure on scoring {} with {} states".format(self.this_word, n_components))
                    continue


            if count > 0:
                # Get the average likelihood score
                avg_logL = total_logL / count
                if self.verbose:
                    print("Average Log {} with {} states".format(avg_logL, n_components))
                if avg_logL > best_logL:
                    best_logL = avg_logL
                    best_n_components = n_components

        # Reset the original XLength values
        self.X, self.lengths = self.hwords[self.this_word]

        # Return the winning model trained on the full training set
        return self.base_model(best_n_components)


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

    #print(Xlengths)
    word = 'BOOK'

    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word,
                       min_n_components=2, max_n_components=15, random_state=14, verbose=True).select()
    end = timeit.default_timer() - start

    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))