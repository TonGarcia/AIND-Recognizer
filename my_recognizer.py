import warnings
from asl_data import SinglesData


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

    # init returned vars
    guesses = []
    probabilities = []

    # iterate the test_set
    for word_id in test_set.get_all_Xlengths():
        # probabilities is a list of dictionaries where each key a word and value is Log Likelihood
        probability_log_likelihoods = {}

        # extract current test_set item based on it id
        current_word_features, current_seq_lengths = test_set.get_item_Xlengths(word_id)

        # calculate LogLikelihoodScore for each word and model, than add it to probabilities list
        for word, model in models.items():
            try:
                # perform score calc
                score = model.score(current_word_features, current_seq_lengths)
                # the key is the a word and it value is the Log Likelihood Score
                probability_log_likelihoods[word] = score
            except:
                # if catch an exception, so it model isn't viable to calc, store it as neg inf score
                probability_log_likelihoods[word] = float("-inf")

        # add it current probability to the probabilities list
        probabilities.append(probability_log_likelihoods)

        # calc the best score
        best_guess_score = max(probability_log_likelihoods, key=probability_log_likelihoods.get)
        # add it best score guess to guesses list, as it follows the test set word_id order
        guesses.append(best_guess_score)

    # return these filled lists
    return probabilities, guesses
