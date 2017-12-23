import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on BIC scores

        # init it bic scores array, filled on the loop
        bic_scores = []
        # sum of it lengths (outside of the loop to improve it perform)
        sum_data_points = sum(self.lengths)

        # BIC score for n between self.min_n_components and self.max_n_components
        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                # Hidden Markov Model
                hmm_model = self.base_model(num_states)
                # logarithm log_likelihood HMM score
                log_likelihood = hmm_model.score(self.X, self.lengths)
                # calc the probability p
                p = (num_states ** 2) + (2 * num_states * sum_data_points) - 1
                # Bayesian information criteria: BIC = -2 * logL + p * logN
                bic_score = (-2 * log_likelihood) + (p * np.log(sum_data_points))
                # add it current bic_score to the bic_scores list
                bic_scores.append(tuple([bic_score, hmm_model]))
            except:
                pass

        # get it min bic_score from bic_scores comparing it first "dictionary" param as the num value
        return min(bic_scores, key = lambda x: x[0])[1] if bic_scores else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on DIC scores

        # Init empty lists
        models = []
        dic_scores = []
        remain_words = []

        # Create a new words list (other_words), removing the current word (self.this_word)
        for word in self.words:
            if word != self.this_word:
                remain_words.append(self.hwords[word])

        # Try to calc. the models iterating the states
        try:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(num_states)
                log_likelihood_original_word = hmm_model.score(self.X, self.lengths)
                models.append((log_likelihood_original_word, hmm_model))
        # Cause Exception if have more params to fit
        # so must catch exception when the model is invalid
        except Exception as e:
            pass

        # Calc the DIC Scores based on the built models above
        for index, model in enumerate(models):
            # current iteration vars
            log_likelihood_original_word, hmm_model = model
            # calc the log for remain words
            log_likelihood_remain_words = np.mean([model[1].score(word[0], word[1]) for word in remain_words])
            # current dic score is the diff between original word & the remain words logs
            dic_score = log_likelihood_original_word - log_likelihood_remain_words
            dic_scores.append(tuple([dic_score, model[1]]))

        # The model is better as greater it DIC score is
        return max(dic_scores, key=lambda x: x[0])[1] if dic_scores else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection using CV

        # init empty objects
        hmm_model = None
        log_likelihood = None

        # init empty lists
        cv_scores = []
        log_likelihoods = []

        # SKLearn KFold: Provides train/test indices to split data in train/test sets.
        # Split dataset into k consecutive folds (without shuffling by default)
        kf = KFold(n_splits=3, shuffle=False, random_state=None)

        # iterate the n components (min & max)
        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                # Check if there is sufficient data to split using KFold
                if len(self.sequences) > 2:
                    # CV loop on training set sequence, for each "folds"
                    # each fold rotated out of the training set is tested by scoring for Cross-Validation (CV)
                    for train_index, test_index in kf.split(self.sequences):
                        # KFold recombining Training sequences split
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        # KFold recombining Testing sequences split
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)

                        # Hidden Markov Model
                        hmm_model = self.base_model(num_states)
                        # Calc it log_likelihood logarithm HMM score
                        log_likelihood = hmm_model.score(X_test, lengths_test)

                # No sufficient data to split using KFold
                else:
                    # Just create the Hidden Markov Model & calc it logarithm HMM score
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)

                log_likelihoods.append(log_likelihood)

                # Find average Log Likelihood of CV fold
                cv_score_avg = np.mean(log_likelihoods)
                cv_scores.append(tuple([cv_score_avg, hmm_model]))

            except Exception as e:
                pass

        # The CV use the max score as the best choice
        return max(cv_scores, key = lambda x: x[0])[1] if cv_scores else None