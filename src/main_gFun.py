import os
from dataset_builder import MultilingualDataset
from learning.transformers import *
from util.evaluation import *
from util.file import exists
from util.results import PolylingualClassificationResults
from util.common import *
from util.parser_options import *

if __name__ == '__main__':
    (op, args) = parser.parse_args()
    dataset = op.dataset
    assert exists(dataset), 'Unable to find file '+str(dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'
    assert op.posteriors or op.supervised or op.pretrained or op.mbert or op.gruViewGenerator, \
        'empty set of document embeddings is not allowed'
    if op.gruViewGenerator:
        assert op.gruWCE or op.gruMUSE, 'Initializing Gated Recurrent embedding layer without ' \
                                                              'explicit initialization of GRU View Generator'

    l2 = op.l2
    dataset_file = os.path.basename(dataset)
    results = PolylingualClassificationResults('../log/' + op.output)
    allprob = 'Prob' if op.allprob else ''

    # renaming arguments to be printed on log
    method_name, dataset_name = get_method_name(dataset, op.posteriors, op.supervised, op.pretrained, op.mbert,
                                                op.gruViewGenerator, op.gruMUSE, op.gruWCE, op.agg, op.allprob)
    print(f'Method: gFun{method_name}\nDataset: {dataset_name}')
    print('-'*50)

    # set zscore range - is slice(0, 0) mean will be equal to 0 and std to 1, thus normalization will have no effect
    standardize_range = slice(0, 0)
    if op.zscore:
        standardize_range = None

    # load dataset
    data = MultilingualDataset.load(dataset)
    data.set_view(languages=['nl', 'it'])   # TODO: DEBUG SETTING
    data.show_dimensions()
    lXtr, lytr = data.training()
    lXte, lyte = data.test()

    # text preprocessing
    tfidfvectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    # feature weighting (for word embeddings average)
    feat_weighting = FeatureWeight(op.feat_weight, agg='mean')

    # document embedding modules aka View Generators
    doc_embedder = DocEmbedderList(aggregation='mean' if op.agg else 'concat')

    # init View Generators
    if op.posteriors:
        """ 
        View Generator (-X): cast document representations encoded via TFIDF into posterior probabilities by means
        of a set of SVM.
        """
        # Check if we already have VG outputs from previous runs
        VG_name = 'X'
        storing_path = f'../dumps/{VG_name}/{dataset_name.split(".")[0]}'
        exist = exists(storing_path)
        doc_embedder.append(PosteriorProbabilitiesEmbedder(first_tier_learner=get_learner(calibrate=True,
                                                                                          kernel='linear',
                                                                                          C=op.set_c),
                                                           l2=l2, storing_path=storing_path))

    if op.supervised:
        """ 
        View Generator (-W): generates document representation via Word-Class-Embeddings.
        Document embeddings are obtained via weighted sum of document's constituent embeddings.
        """
        VG_name = 'W'
        storing_path = f'../dumps/{VG_name}/{dataset_name.split(".")[0]}'
        exist = exists(storing_path)
        wce = WordClassEmbedder(max_label_space=op.max_labels_S, l2=l2, featureweight=feat_weighting, sif=op.sif)
        if op.allprob:
            wce = FeatureSet2Posteriors(wce, method_id=VG_name, requires_tfidf=True, l2=l2, storing_path=storing_path)
        doc_embedder.append(wce)

    if op.pretrained:
        """
        View Generator (-M): generates document representation via MUSE embeddings (Fasttext multilingual word 
        embeddings). Document embeddings are obtained via weighted sum of document's constituent embeddings.
        """
        VG_name = 'M'
        storing_path = f'../dumps/{VG_name}/{dataset_name.split(".")[0]}'
        exist = exists(storing_path)
        muse = MuseEmbedder(op.we_path, l2=l2, featureweight=feat_weighting, sif=op.sif)
        if op.allprob:
            muse = FeatureSet2Posteriors(muse, method_id=VG_name, requires_tfidf=True, l2=l2, storing_path=storing_path)
        doc_embedder.append(muse)

    if op.gruViewGenerator:
        """
        View Generator (-G): generates document embedding by means of a Gated Recurrent Units. The model can be 
        initialized with different (multilingual/aligned) word representations (e.g., MUSE, WCE, ecc.,). 
        Output dimension is (n_docs, 512). If --allprob output will be casted to posterior prob space via SVM.
        """
        VG_name = 'G'
        VG_name += '_muse' if op.gruMUSE else ''
        VG_name += '_wce' if op.gruWCE else ''
        storing_path = f'../dumps/{VG_name}/{dataset_name.split(".")[0]}'
        rnn_embedder = RecurrentEmbedder(pretrained=op.gruMUSE, supervised=op.gruWCE, multilingual_dataset=data,
                                         options=op, model_path=op.gru_path)
        if op.allprob:
            rnn_embedder = FeatureSet2Posteriors(rnn_embedder, method_id=VG_name, requires_tfidf=False,
                                                 storing_path=storing_path)
        doc_embedder.append(rnn_embedder)

    if op.mbert:
        """
        View generator (-B): generates document embedding via mBERT model. 
        """
        VG_name = 'B'
        storing_path = f'../dumps/{VG_name}/{dataset_name.split(".")[0]}'

        mbert = MBertEmbedder(path_to_model=op.bert_path, nC=data.num_categories())
        if op.allprob:
            mbert = FeatureSet2Posteriors(mbert, method_id=VG_name, l2=l2, storing_path=storing_path)
        doc_embedder.append(mbert)

    # metaclassifier
    meta_parameters = None if op.set_c != -1 else [{'C': [1, 1e3, 1e2, 1e1, 1e-1]}]
    meta = MetaClassifier(meta_learner=get_learner(calibrate=False, kernel='rbf', C=op.set_c),
                          meta_parameters=get_params(op.optimc), standardize_range=standardize_range)

    # ensembling the modules
    classifier = Funnelling(vectorizer=tfidfvectorizer, first_tier=doc_embedder, meta=meta)

    print('\n# Fitting Funnelling Architecture...')
    tinit = time.time()
    classifier.fit(lXtr, lytr)
    time = time.time()-tinit

    print('\n# Evaluating ...')
    l_eval = evaluate_method(classifier, lXte, lyte)

    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
        results.add_row(method='MultiModal',
                        learner='SVM',
                        optimp=op.optimc,
                        sif=op.sif,
                        zscore=op.zscore,
                        l2=op.l2,
                        wescaler=op.feat_weight,
                        pca=op.max_labels_S,
                        id=method_name,
                        dataset=dataset_name,
                        time=time,
                        lang=lang,
                        macrof1=macrof1,
                        microf1=microf1,
                        macrok=macrok,
                        microk=microk,
                        notes='')
    print('Averages: MF1, mF1, MK, mK', np.round(np.mean(np.array(metrics), axis=0), 3))
