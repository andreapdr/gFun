import os
from dataset_builder import MultilingualDataset
# from learning.learners import *
# from learning.learners import FunnellingMultimodal
from learning.transformers import Funnelling, PosteriorProbabilitiesEmbedder, MetaClassifier, \
    TfidfVectorizerMultilingual, DocEmbedderList, WordClassEmbedder, MuseEmbedder, FeatureSet2Posteriors, Voting
from util.evaluation import *
from optparse import OptionParser
from util.file import exists
from util.results import PolylingualClassificationResults
from sklearn.svm import SVC
from util.util import get_learner, get_params
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

parser = OptionParser()

# parser.add_option("-d", "--dataset", dest="dataset",
#                   help="Path to the multilingual dataset processed and stored in .pickle format",
#                   default="../rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle")

parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results/results.csv')

parser.add_option("-P", "--probs", dest="posteriors", action='store_true',
                  help="Add posterior probabilities to the document embedding representation", default=False)

parser.add_option("-S", "--supervised", dest="supervised", action='store_true',
                  help="Add supervised (Word-Class Embeddings) to the document embedding representation", default=False)

parser.add_option("-U", "--pretrained", dest="pretrained", action='store_true',
                  help="Add pretrained MUSE embeddings to the document embedding representation", default=False)

parser.add_option("-w", "--we-path", dest="we_path",
                  help="Path to the MUSE polylingual word embeddings", default='../embeddings')

parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)

parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimize hyperparameters", default=False)

parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)

parser.add_option("-p", "--pca", dest="max_labels_S", type=int,
                  help="If smaller than number of target classes, PCA will be applied to supervised matrix. ",
                  default=300)

parser.add_option("-r", "--remove-pc", dest="sif", action='store_true',
                  help="Remove common component when computing dot product of word embedding matrices", default=False)

# parser.add_option("-u", "--upca", dest="max_labels_U", type=int,
#                   help="If smaller than Unsupervised Dimension, PCA will be applied to unsupervised matrix."
#                        " If set to 0 it will automatically search for the best number of components", default=300)

# parser.add_option("-a", dest="post_pca",
#                   help="If set to True, will apply PCA to the z-space (posterior probabilities stacked along with "
#                        "embedding space", default=False)


def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1, gamma='auto')


def get_params(dense=False):
    if not op.optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    kernel = 'rbf' if dense else 'linear'
    return [{'kernel': [kernel], 'C': c_range, 'gamma':['auto']}]

#######################################################################################################################


if __name__ == '__main__':
    (op, args) = parser.parse_args()

    assert len(args)==1, 'required argument "datapath" missing (path to the pickled dataset)'
    dataset = args[0]

    assert exists(dataset), 'Unable to find file '+str(dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'
    assert op.posteriors or op.supervised or op.pretrained, 'empty set of document embeddings is not allowed'

    dataset_file = os.path.basename(dataset)

    results = PolylingualClassificationResults(op.output)

    data = MultilingualDataset.load(dataset)
    data.show_dimensions()

    lXtr, lytr = data.training()
    lXte, lyte = data.test()

    meta_parameters = None if op.set_c != -1 else [{'C': [1, 1e3, 1e2, 1e1, 1e-1]}]

    # result_id = f'{dataset_file}_Prob{op.posteriors}_WCE{op.supervised}(PCA{op.max_labels_S})_MUSE{op.pretrained}{"_optimC" if op.optimc else ""}'
    result_id = f'{dataset_file}_ProbPost={op.posteriors}_WCE={op.supervised}(PCA={op.max_labels_S})_' \
                f'MUSE={op.pretrained}_weight={"todo"}_l2={"todo"}_zscore={"todo"}{"_optimC" if op.optimc else ""}'
    print(f'{result_id}')

    # text preprocessing
    tfidfvectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    lXtr = tfidfvectorizer.fit_transform(lXtr, lytr)
    lXte = tfidfvectorizer.transform(lXte)
    lV = tfidfvectorizer.vocabulary()

    classifiers = []
    if op.posteriors:
        classifiers.append(PosteriorProbabilitiesEmbedder(first_tier_learner=get_learner(calibrate=True), first_tier_parameters=None))
    if op.supervised:
        classifiers.append(FeatureSet2Posteriors(WordClassEmbedder(max_label_space=op.max_labels_S)))
    if op.pretrained:
        classifiers.append(FeatureSet2Posteriors(MuseEmbedder(op.we_path, lV=lV)))

    classifier = Voting(*classifiers)

    print('# Fitting ...')
    classifier.fit(lXtr, lytr)

    print('\n# Evaluating ...')
    l_eval = evaluate_method(classifier, lXte, lyte)

    # renaming arguments to be printed on log
    _id = ''
    _id_conf = [op.posteriors, op.supervised, op.pretrained]
    _id_name = ['+P', '+W', '+M']
    for i, conf in enumerate(_id_conf):
        if conf:
            _id += _id_name[i]
    _id = _id.lstrip('+')
    _dataset_path = dataset.split('/')[-1].split('_')
    dataset_id = _dataset_path[0] + _dataset_path[-1]

    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
        results.add_row(method='Voting',
                        learner='svm',
                        optimp=op.optimc,
                        sif=op.sif,
                        zscore='todo',
                        l2='todo',
                        wescaler='todo',
                        pca=op.max_labels_S,
                        id=_id,
                        dataset=dataset_id,
                        time='todo',
                        lang=lang,
                        macrof1=macrof1,
                        microf1=microf1,
                        macrok=macrok,
                        microk=microk,
                        notes='')
    print('Averages: MF1, mF1, MK, mK', np.mean(np.array(metrics), axis=0))
