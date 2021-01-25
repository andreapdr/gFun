from argparse import ArgumentParser
from funnelling import *
from view_generators import *
from data.dataset_builder import MultilingualDataset
from util.common import MultilingualIndex, get_params
from util.evaluation import evaluate
from util.results_csv import CSVlog
from time import time


def main(args):
    OPTIMC = True # TODO
    N_JOBS = 8
    print('Running refactored...')

    # _DATASET = '/homenfs/a.pedrotti1/datasets/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle'
    # EMBEDDINGS_PATH = '/homenfs/a.pedrotti1/embeddings/MUSE'

    _DATASET = '/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle'
    EMBEDDINGS_PATH = '/home/andreapdr/gfun/embeddings'
    data = MultilingualDataset.load(_DATASET)
    data.set_view(languages=['it', 'fr'])
    lX, ly = data.training()
    lXte, lyte = data.test()

    # Init multilingualIndex - mandatory when deploying Neural View Generators...
    multilingualIndex = MultilingualIndex()
    lMuse = MuseLoader(langs=sorted(lX.keys()), cache=EMBEDDINGS_PATH)
    multilingualIndex.index(lX, ly, lXte, lyte, l_pretrained_vocabulary=lMuse.vocabulary())

    embedder_list = []
    if args.X:
        posteriorEmbedder = VanillaFunGen(base_learner=get_learner(calibrate=True), n_jobs=N_JOBS)
        embedder_list.append(posteriorEmbedder)

    if args.M:
        museEmbedder = MuseGen(muse_dir=EMBEDDINGS_PATH, n_jobs=N_JOBS)
        embedder_list.append(museEmbedder)

    if args.W:
        wceEmbedder = WordClassGen(n_jobs=N_JOBS)
        embedder_list.append(wceEmbedder)

    if args.G:
        rnnEmbedder = RecurrentGen(multilingualIndex, pretrained_embeddings=lMuse, wce=False, batch_size=256,
                                   nepochs=250, gpus=args.gpus, n_jobs=N_JOBS)
        embedder_list.append(rnnEmbedder)

    if args.B:
        bertEmbedder = BertGen(multilingualIndex, batch_size=4, nepochs=1, gpus=args.gpus, n_jobs=N_JOBS)
        embedder_list.append(bertEmbedder)

    # Init DocEmbedderList
    docEmbedders = DocEmbedderList(embedder_list=embedder_list, probabilistic=True)
    meta_parameters = None if not OPTIMC else [{'C': [1, 1e3, 1e2, 1e1, 1e-1]}]
    meta = MetaClassifier(meta_learner=get_learner(calibrate=False, kernel='rbf', C=meta_parameters),
                          meta_parameters=get_params(optimc=True))

    # Init Funnelling Architecture
    gfun = Funnelling(first_tier=docEmbedders, meta_classifier=meta)

    # Training ---------------------------------------
    print('\n[Training Generalized Funnelling]')
    time_init = time()
    time_tr = time()
    gfun.fit(lX, ly)
    time_tr = round(time() - time_tr, 3)
    print(f'Training completed in {time_tr} seconds!')

    # Testing ----------------------------------------
    print('\n[Testing Generalized Funnelling]')
    time_te = time()
    ly_ = gfun.predict(lXte)
    l_eval = evaluate(ly_true=ly, ly_pred=ly_)
    time_te = round(time() - time_te, 3)
    print(f'Testing completed in {time_te} seconds!')

    # Logging ---------------------------------------
    print('\n[Results]')
    results = CSVlog('test_log.csv')
    metrics = []
    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        print(f'Lang {lang}: macro-F1 = {macrof1:.3f} micro-F1 = {microf1:.3f}')
        results.add_row(method='gfun',
                        setting='TODO',
                        sif='True',
                        zscore='True',
                        l2='True',
                        dataset='TODO',
                        time_tr=time_tr,
                        time_te=time_te,
                        lang=lang,
                        macrof1=macrof1,
                        microf1=microf1,
                        macrok=macrok,
                        microk=microk,
                        notes='')
    print('Averages: MF1, mF1, MK, mK', np.round(np.mean(np.array(metrics), axis=0), 3))

    overall_time = round(time() - time_init, 3)
    exit(f'\nExecuted in: {overall_time } seconds!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--X')
    parser.add_argument('--M')
    parser.add_argument('--W')
    parser.add_argument('--G')
    parser.add_argument('--B')
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()
    main(args)
