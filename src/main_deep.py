from optparse import OptionParser
from util.results import PolylingualClassificationResults
from dataset_builder import MultilingualDataset
from keras.preprocessing.text import Tokenizer
from learning.learners import MonolingualNetSvm
from sklearn.svm import SVC
import pickle

parser = OptionParser()

parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format",
                  default="/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle")

parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimize hyperparameters", default=False)

parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)

(op, args) = parser.parse_args()


###################################################################################################################

def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1, class_weight='balanced', gamma='auto')


def get_params(dense=False):
    if not op.optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    kernel = 'rbf' if dense else 'linear'
    return [{'kernel': [kernel], 'C': c_range, 'gamma':['auto']}]


# PREPROCESS TEXT AND SAVE IT ... both for SVM and NN
def preprocess_data(lXtr, lXte, lytr, lyte):
    tokenized_tr = dict()
    tokenized_te = dict()
    for lang in lXtr.keys():
        alltexts = ' '.join(lXtr[lang])
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(alltexts.split(' '))
        tokenizer.oov_token = len(tokenizer.word_index)+1
        # dumping train set
        sequences_tr = tokenizer.texts_to_sequences(lXtr[lang])
        tokenized_tr[lang] = (tokenizer.word_index, sequences_tr, lytr[lang])
        # dumping test set
        sequences_te = tokenizer.texts_to_sequences(lXte[lang])
        tokenized_te[lang] = (tokenizer.word_index, sequences_te, lyte[lang])

    with open('/home/andreapdr/CLESA/preprocessed_dataset_nn/rcv1-2_train.pickle', 'wb') as f:
        pickle.dump(tokenized_tr, f)

    with open('/home/andreapdr/CLESA/preprocessed_dataset_nn/rcv1-2_test.pickle', 'wb') as f:
        pickle.dump(tokenized_tr, f)

    print('Successfully dumped data')

# def load_preprocessed():
#     with open('/home/andreapdr/CLESA/preprocessed_dataset_nn/rcv1-2_train.pickle', 'rb') as f:
#         return pickle.load(f)
#
# def build_embedding_matrix(lang, word_index):
#     type = 'MUSE'
#     path = '/home/andreapdr/CLESA/'
#     MUSE = EmbeddingsAligned(type, path, lang, word_index.keys())
#     return MUSE


########## MAIN #################################################################################################

if __name__ == '__main__':
    results = PolylingualClassificationResults('./results/NN_FPEC_results.csv')
    data = MultilingualDataset.load(op.dataset)
    lXtr, lytr = data.training()
    lXte, lyte = data.test()

    if op.set_c != -1:
        meta_parameters = None
    else:
        meta_parameters = [{'C': [1e3, 1e2, 1e1, 1, 1e-1]}]

    test_architecture = MonolingualNetSvm(lXtr,
                                          lytr,
                                          first_tier_learner=get_learner(calibrate=True),
                                          first_tier_parameters=None,
                                          n_jobs=1)

    test_architecture.fit()
