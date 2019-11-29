import os, sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.file import exists
from util.results import PolylingualClassificationResults
from sklearn.svm import SVC


parser = OptionParser()

parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")

parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results/results.csv')

parser.add_option("-e", "--mode-embed", dest="mode_embed",
                  help="Set the embedding to be used [none, pretrained, supervised, both]", type=str, default='none')

parser.add_option("-w", "--we-path", dest="we_path",
                  help="Path to the polylingual word embeddings", default='../embeddings/')

parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)

parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimices hyperparameters", default=False)

parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)


def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1, class_weight='balanced')


def get_params(dense=False):    # TODO kernel function could be useful for meta-classifier
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

    # data.set_view(languages=['en','it'], categories=list(range(10)))
    lXtr, lytr = data.training()
    lXte, lyte = data.test()


    if op.set_c != -1:
        meta_parameters = None
    else:
        meta_parameters = [{'C': [1e3, 1e2, 1e1, 1, 1e-1]}]

    # Embeddings and WCE config
    _available_mode = ['none', 'unsupervised', 'supervised', 'both']
    assert op.mode_embed in _available_mode , f'{op.mode_embed} not in {_available_mode}'

    if op.mode_embed == 'none':
        config = {'unsupervised': False,
                    'supervised': False}
        _config_id = 'None'
    elif op.mode_embed == 'unsupervised':
        config = {'unsupervised': True,
                  'supervised': False}
        _config_id = 'M'
    elif op.mode_embed == 'supervised':
        config = {'unsupervised': False,
                  'supervised': True}
        _config_id = 'F'
    elif op.mode_embed == 'both':
        config = {'unsupervised': True,
                  'supervised': True}
        _config_id = 'M_and_F'

    result_id = dataset_file + 'PolyEmbedd_andrea_' + _config_id + ('_optimC' if op.optimc else '')

    print(f'### PolyEmbedd_andrea_{_config_id}\n')
    classifier = AndreaCLF(op.we_path,
                           config,
                           first_tier_learner=get_learner(calibrate=True),
                           meta_learner=get_learner(calibrate=False, kernel='rbf'),
                           first_tier_parameters=get_params(dense=False),
                           meta_parameters=get_params(dense=True),
                           n_jobs=op.n_jobs)

    print('# Fitting ...')
    classifier.fit(lXtr, lytr)

    print('# Evaluating ...')
    l_eval = evaluate_method(classifier, lXte, lyte)

    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        results.add_row(result_id, 'PolyEmbed_andrea', 'svm', _config_id, op.optimc, op.dataset.split('/')[-1],
                        'not_binary', 'not_ablation', classifier.time, lang, macrof1, microf1, macrok, microk, 'nope')
    print('Averages: MF1, mF1, MK, mK', np.mean(np.array(metrics), axis=0))
