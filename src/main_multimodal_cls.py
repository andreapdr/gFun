import os
from dataset_builder import MultilingualDataset
from learning.transformers import *
from util.evaluation import *
from optparse import OptionParser
from util.file import exists
from util.results import PolylingualClassificationResults
from sklearn.svm import SVC


parser = OptionParser(usage="usage: %prog datapath [options]")

parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results/results.csv')

parser.add_option("-P", "--posteriors", dest="posteriors", action='store_true',
                  help="Add posterior probabilities to the document embedding representation", default=False)

parser.add_option("-S", "--supervised", dest="supervised", action='store_true',
                  help="Add supervised (Word-Class Embeddings) to the document embedding representation", default=False)

parser.add_option("-U", "--pretrained", dest="pretrained", action='store_true',
                  help="Add pretrained MUSE embeddings to the document embedding representation", default=False)

parser.add_option("--nol2", dest="nol2", action='store_true',
                  help="Deactivates l2 normalization as a post-processing for the document embedding views", default=False)

parser.add_option("--allprob", dest="allprob", action='store_true',
                  help="All views are generated as posterior probabilities. This affects the supervised and pretrained "
                       "embeddings, for which a calibrated classifier is generated, which generates the posteriors", default=False)

parser.add_option("--feat-weight", dest="feat_weight",
                  help="Term weighting function to weight the averaged embeddings", type=str,  default='tfidf')

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



def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1, gamma='auto')

#######################################################################################################################


if __name__ == '__main__':
    (op, args) = parser.parse_args()

    assert len(args)==1, 'required argument "datapath" missing (path to the pickled dataset)'
    dataset = args[0]
    assert exists(dataset), 'Unable to find file '+str(dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'
    assert op.posteriors or op.supervised or op.pretrained, 'empty set of document embeddings is not allowed'
    l2=(op.nol2==False)

    dataset_file = os.path.basename(dataset)

    results = PolylingualClassificationResults(op.output)
    allprob='Prob' if op.allprob else ''
    result_id = f'{dataset_file}_ProbPost={op.posteriors}_{allprob}WCE={op.supervised}(PCA={op.max_labels_S})_{allprob}' \
        f'MUSE={op.pretrained}_weight={op.feat_weight}_l2={l2}{"_optimC" if op.optimc else ""}'
    print(f'{result_id}')

    data = MultilingualDataset.load(dataset)
    data.show_dimensions()
    lXtr, lytr = data.training()
    lXte, lyte = data.test()

    # text preprocessing
    tfidfvectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    # feature weighting (for word embeddings average)
    feat_weighting = FeatureWeight(op.feat_weight, agg='mean')

    # # document embedding modules
    doc_embedder = DocEmbedderList(aggregation='concat')
    if op.posteriors:
        doc_embedder.append(PosteriorProbabilitiesEmbedder(first_tier_learner=get_learner(calibrate=True, kernel='linear'), l2=l2))
    if op.supervised:
        wce = WordClassEmbedder(max_label_space=op.max_labels_S, l2=l2, featureweight=feat_weighting)
        if op.allprob:
            wce = FeatureSet2Posteriors(wce, l2=l2)
        doc_embedder.append(wce)
    if op.pretrained:
        muse = MuseEmbedder(op.we_path, l2=l2, featureweight=feat_weighting)
        if op.allprob:
            muse = FeatureSet2Posteriors(muse, l2=l2)
        doc_embedder.append(muse)

    # metaclassifier
    meta_parameters = None if op.set_c != -1 else [{'C': [1, 1e3, 1e2, 1e1, 1e-1]}]
    meta = MetaClassifier(meta_learner=get_learner(calibrate=False, kernel='rbf'), meta_parameters=meta_parameters)

    # ensembling the modules
    classifier = Funnelling(vectorizer=tfidfvectorizer, first_tier=doc_embedder, meta=meta)

    print('# Fitting ...')
    classifier.fit(lXtr, lytr)

    print('\n# Evaluating ...')
    l_eval = evaluate_method(classifier, lXte, lyte)

    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
        # results.add_row('PolyEmbed_andrea', 'svm', _config_id, config['we_type'],
        #                 (config['max_label_space'], classifier.best_components),
        #                 config['dim_reduction_unsupervised'], op.optimc, dataset.split('/')[-1], classifier.time,
        #                 lang, macrof1, microf1, macrok, microk, '')
    print('Averages: MF1, mF1, MK, mK', np.mean(np.array(metrics), axis=0))
