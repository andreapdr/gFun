import os
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.file import exists
from util.results import PolylingualClassificationResults
from sklearn.svm import SVC


parser = OptionParser()

parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format",
                  default="/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle")

parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results/results.csv')

parser.add_option("-e", "--mode-embed", dest="mode_embed",
                  help="Set the embedding to be used [none, unsupervised, supervised, both]", type=str, default='none')

parser.add_option("-w", "--we-path", dest="we_path",
                  help="Path to the polylingual word embeddings", default='/home/andreapdr/CLESA/')

parser.add_option('-t', "--we-type", dest="we_type", help="Aligned embeddings to use [FastText, MUSE]", type=str,
                  default='MUSE')

parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)

parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimize hyperparameters", default=False)

parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)

parser.add_option("-p", "--pca", dest="max_labels", type=int,
                  help="If less than number of target classes, will apply PCA to supervised matrix. If set to 0 it"
                       " will automatically search for the best number of components", default=300)

parser.add_option("-u", "--upca", dest="max_labels_U", type=int,
                  help="If smaller than Unsupervised Dimension, will apply PCA to unsupervised matrix. If set to 0 it"
                       " will automatically search for the best number of components", default=300)

parser.add_option("-l", dest="lang", type=str)


def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1, class_weight='balanced', gamma='auto')


def get_params(dense=False):
    if not op.optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    kernel = 'rbf' if dense else 'linear'
    return [{'kernel': [kernel], 'C': c_range, 'gamma':['auto']}]

#######################################################################################################################


if __name__ == '__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'

    dataset_file = os.path.basename(op.dataset)

    results = PolylingualClassificationResults(op.output)

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    data.set_view(languages=['en','it', 'pt', 'sv'], categories=list(range(10)))
    # data.set_view(languages=[op.lang])
    # data.set_view(categories=list(range(10)))
    lXtr, lytr = data.training()
    lXte, lyte = data.test()


    if op.set_c != -1:
        meta_parameters = None
    else:
        meta_parameters = [{'C': [1e3, 1e2, 1e1, 1, 1e-1]}]

    # Embeddings and WCE config
    _available_mode = ['none', 'unsupervised', 'supervised', 'both']
    _available_type = ['MUSE', 'FastText']
    assert op.mode_embed in _available_mode, f'{op.mode_embed} not in {_available_mode}'
    assert op.we_type in _available_type, f'{op.we_type} not in {_available_type}'

    if op.mode_embed == 'none':
        config = {'unsupervised': False,
                  'supervised': False,
                  'we_type': None}
        _config_id = 'None'
    elif op.mode_embed == 'unsupervised':
        config = {'unsupervised': True,
                  'supervised': False,
                  'we_type': op.we_type}
        _config_id = 'M'
    elif op.mode_embed == 'supervised':
        config = {'unsupervised': False,
                  'supervised': True,
                  'we_type': None}
        _config_id = 'F'
    elif op.mode_embed == 'both':
        config = {'unsupervised': True,
                  'supervised': True,
                  'we_type': op.we_type}
        _config_id = 'M_and_F'

    ##### TODO - config dict is redundant - we have already op argparse ...
    config['reduction'] = 'PCA'
    config['max_label_space'] = op.max_labels
    config['dim_reduction_unsupervised'] = op.max_labels_U
    # config['plot_covariance_matrices'] = True

    result_id = dataset_file + 'PolyEmbedd_andrea_' + _config_id + ('_optimC' if op.optimc else '')

    print(f'### PolyEmbedd_andrea_{_config_id}\n')
    classifier = AndreaCLF(we_path=op.we_path,
                           config=config,
                           first_tier_learner=get_learner(calibrate=True),
                           meta_learner=get_learner(calibrate=False, kernel='rbf'),
                           first_tier_parameters=get_params(dense=False),
                           meta_parameters=get_params(dense=True),
                           n_jobs=op.n_jobs)

    print('# Fitting ...')
    classifier.fit(lXtr, lytr)

    print('\n# Evaluating ...')
    l_eval = evaluate_method(classifier, lXte, lyte)

    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        results.add_row(result_id, 'PolyEmbed_andrea', 'svm', _config_id, config['we_type'], op.optimc, op.dataset.split('/')[-1],
                        classifier.time, lang, macrof1, microf1, macrok, microk, '')
    print('Averages: MF1, mF1, MK, mK', np.mean(np.array(metrics), axis=0))
