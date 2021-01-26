from argparse import ArgumentParser

from src.data.dataset_builder import MultilingualDataset
from src.funnelling import *
from src.util.common import MultilingualIndex, get_params, get_method_name
from src.util.evaluation import evaluate
from src.util.results_csv import CSVlog
from src.view_generators import *


def main(args):
    assert args.post_embedder or args.muse_embedder or args.wce_embedder or args.gru_embedder or args.bert_embedder, \
        'empty set of document embeddings is not allowed!'

    print('Running generalized funnelling...')

    data = MultilingualDataset.load(args.dataset)
    data.set_view(languages=['it', 'fr'])
    data.show_dimensions()
    lX, ly = data.training()
    lXte, lyte = data.test()

    # Init multilingualIndex - mandatory when deploying Neural View Generators...
    if args.gru_embedder or args.bert_embedder:
        multilingualIndex = MultilingualIndex()
        lMuse = MuseLoader(langs=sorted(lX.keys()), cache=args.muse_dir)
        multilingualIndex.index(lX, ly, lXte, lyte, l_pretrained_vocabulary=lMuse.vocabulary())

    # Init ViewGenerators and append them to embedder_list
    embedder_list = []
    if args.post_embedder:
        posteriorEmbedder = VanillaFunGen(base_learner=get_learner(calibrate=True), n_jobs=args.n_jobs)
        embedder_list.append(posteriorEmbedder)

    if args.muse_embedder:
        museEmbedder = MuseGen(muse_dir=args.muse_dir, n_jobs=args.n_jobs)
        embedder_list.append(museEmbedder)

    if args.wce_embedder:
        wceEmbedder = WordClassGen(n_jobs=args.n_jobs)
        embedder_list.append(wceEmbedder)

    if args.gru_embedder:
        rnnEmbedder = RecurrentGen(multilingualIndex, pretrained_embeddings=lMuse, wce=args.gru_wce, batch_size=256,
                                   nepochs=args.nepochs, gpus=args.gpus, n_jobs=args.n_jobs)
        embedder_list.append(rnnEmbedder)

    if args.bert_embedder:
        bertEmbedder = BertGen(multilingualIndex, batch_size=4, nepochs=10, gpus=args.gpus, n_jobs=args.n_jobs)
        embedder_list.append(bertEmbedder)

    # Init DocEmbedderList (i.e., first-tier learners or view generators) and metaclassifier
    docEmbedders = DocEmbedderList(embedder_list=embedder_list, probabilistic=True)
    meta = MetaClassifier(meta_learner=get_learner(calibrate=False, kernel='rbf'),
                          meta_parameters=get_params(optimc=args.optimc))

    # Init Funnelling Architecture
    gfun = Funnelling(first_tier=docEmbedders, meta_classifier=meta)

    # Training ---------------------------------------
    print('\n[Training Generalized Funnelling]')
    time_init = time.time()
    gfun.fit(lX, ly)
    time_tr = round(time.time() - time_init, 3)
    print(f'Training completed in {time_tr} seconds!')

    # Testing ----------------------------------------
    print('\n[Testing Generalized Funnelling]')
    time_te = time.time()
    ly_ = gfun.predict(lXte)
    l_eval = evaluate(ly_true=lyte, ly_pred=ly_)
    time_te = round(time.time() - time_te, 3)
    print(f'Testing completed in {time_te} seconds!')

    # Logging ---------------------------------------
    print('\n[Results]')
    results = CSVlog(args.csv_dir)
    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print(f'Lang {lang}: macro-F1 = {macrof1:.3f} micro-F1 = {microf1:.3f}')
        if results is not None:
            _id, _dataset = get_method_name(args)
            results.add_row(method='gfun',
                            setting=_id,
                            optimc=args.optimc,
                            sif='True',
                            zscore='True',
                            l2='True',
                            dataset=_dataset,
                            time_tr=time_tr,
                            time_te=time_te,
                            lang=lang,
                            macrof1=macrof1,
                            microf1=microf1,
                            macrok=macrok,
                            microk=microk,
                            notes='')
    print('Averages: MF1, mF1, MK, mK', np.round(np.mean(np.array(metrics), axis=0), 3))

    overall_time = round(time.time() - time_init, 3)
    exit(f'\nExecuted in: {overall_time} seconds!')


if __name__ == '__main__':
    parser = ArgumentParser(description='Run generalized funnelling, A. Moreo, A. Pedrotti and F. Sebastiani')

    parser.add_argument('dataset', help='Path to the dataset')

    parser.add_argument('-o', '--output', dest='csv_dir',
                        help='Result file (default ../csv_logs/gfun/gfun_results.csv)', type=str,
                        default='../csv_logs/gfun/gfun_results.csv')

    parser.add_argument('-x', '--post_embedder', dest='post_embedder', action='store_true',
                        help='deploy posterior probabilities embedder to compute document embeddings',
                        default=False)

    parser.add_argument('-w', '--wce_embedder', dest='wce_embedder', action='store_true',
                        help='deploy (supervised) Word-Class embedder to the compute document embeddings',
                        default=False)

    parser.add_argument('-m', '--muse_embedder', dest='muse_embedder', action='store_true',
                        help='deploy (pretrained) MUSE embedder to compute document embeddings',
                        default=False)

    parser.add_argument('-b', '--bert_embedder', dest='bert_embedder', action='store_true',
                        help='deploy multilingual Bert to compute document embeddings',
                        default=False)

    parser.add_argument('-g', '--gru_embedder', dest='gru_embedder', action='store_true',
                        help='deploy a GRU in order to compute document embeddings',
                        default=False)

    parser.add_argument('-c', '--c_optimize', dest='optimc', action='store_true',
                        help='Optimize SVMs C hyperparameter',
                        default=False)

    parser.add_argument('-n', '--nepochs', dest='nepochs', type=int,
                        help='Number of max epochs to train Recurrent embedder (i.e., -g)')

    parser.add_argument('-j', '--n_jobs', dest='n_jobs', type=int,
                        help='Number of parallel jobs (default is -1, all)',
                        default=-1)

    parser.add_argument('--muse_dir', dest='muse_dir', type=str,
                        help='Path to the MUSE polylingual word embeddings (default ../embeddings)',
                        default='../embeddings')

    parser.add_argument('--gru_wce', dest='gru_wce', action='store_true',
                        help='Deploy WCE embedding as embedding layer of the GRU View Generator',
                        default=False)

    parser.add_argument('--gru_dir', dest='gru_dir', type=str,
                        help='Set the path to a pretrained GRU model (i.e., -g view generator)',
                        default=None)

    parser.add_argument('--bert_dir', dest='bert_dir', type=str,
                        help='Set the path to a pretrained mBERT model (i.e., -b view generator)',
                        default=None)

    parser.add_argument('--gpus', help='specifies how many GPUs to use per node',
                        default=None)

    args = parser.parse_args()
    main(args)
