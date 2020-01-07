import os
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.file import exists
from util.results import PolylingualClassificationResults
from util.util import get_learner, get_params

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

parser.add_option("-p", "--pca", dest="max_labels_S", type=int,
                  help="If smaller than number of target classes, PCA will be applied to supervised matrix. "
                       "If set to 0 it will automatically search for the best number of components. "
                       "If set to -1 it will apply PCA to the vstacked supervised matrix (PCA dim set to 50 atm)",
                  default=300)

parser.add_option("-u", "--upca", dest="max_labels_U", type=int,
                  help="If smaller than Unsupervised Dimension, PCA will be applied to unsupervised matrix."
                       " If set to 0 it will automatically search for the best number of components", default=300)

parser.add_option("-l", dest="lang", type=str)

if __name__ == '__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'

    dataset_file = os.path.basename(op.dataset)

    results = PolylingualClassificationResults('./results/PLE_results.csv')

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    # data.set_view(languages=['en','it', 'pt', 'sv'], categories=list(range(10)))
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
        _config_id = 'M+F'

    config['reduction'] = 'PCA'
    config['max_label_space'] = op.max_labels_S
    config['dim_reduction_unsupervised'] = op.max_labels_U
    # config['post_pca'] = op.post_pca
    # config['plot_covariance_matrices'] = True

    result_id = dataset_file + 'MLE_andrea' + _config_id + ('_optimC' if op.optimc else '')

    ple = PolylingualEmbeddingsClassifier(wordembeddings_path='/home/andreapdr/CLESA/',
                                          config = config,
                                          learner=get_learner(calibrate=False),
                                          c_parameters=get_params(dense=False),
                                          n_jobs=op.n_jobs)

    print('# Fitting ...')
    ple.fit(lXtr, lytr)

    print('# Evaluating ...')
    ple_eval = evaluate_method(ple, lXte, lyte)

    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = ple_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        results.add_row('MLE', 'svm', _config_id, config['we_type'],
                        'no','no', op.optimc, op.dataset.split('/')[-1], ple.time,
                        lang, macrof1, microf1, macrok, microk, '')
    print('Averages: MF1, mF1, MK, mK', np.mean(np.array(metrics), axis=0))
