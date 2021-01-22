from argparse import ArgumentParser
from util.embeddings_manager import MuseLoader
from view_generators import RecurrentGen, BertGen
from data.dataset_builder import MultilingualDataset
from util.common import MultilingualIndex
from time import time


def main(args):
    N_JOBS = 8
    print('Running refactored...')

    # _DATASET = '/homenfs/a.pedrotti1/datasets/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle'
    # EMBEDDINGS_PATH = '/homenfs/a.pedrotti1/embeddings/MUSE'

    _DATASET = '/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle'
    EMBEDDINGS_PATH = '/home/andreapdr/gfun/embeddings'
    data = MultilingualDataset.load(_DATASET)
    # data.set_view(languages=['it', 'fr'])
    lX, ly = data.training()
    lXte, lyte = data.test()

    # Init multilingualIndex - mandatory when deploying Neural View Generators...
    multilingualIndex = MultilingualIndex()
    lMuse = MuseLoader(langs=sorted(lX.keys()), cache=EMBEDDINGS_PATH)
    multilingualIndex.index(lX, ly, lXte, lyte, l_pretrained_vocabulary=lMuse.vocabulary())

    # gFun = VanillaFunGen(base_learner=get_learner(calibrate=True), n_jobs=N_JOBS)
    # gFun = MuseGen(muse_dir='/home/andreapdr/funneling_pdr/embeddings', n_jobs=N_JOBS)
    # gFun = WordClassGen(n_jobs=N_JOBS)
    gFun = RecurrentGen(multilingualIndex, pretrained_embeddings=lMuse, wce=False, batch_size=256,
                        nepochs=50, gpus=args.gpus, n_jobs=N_JOBS)
    # gFun = BertGen(multilingualIndex, batch_size=4, nepochs=10, gpus=args.gpus, n_jobs=N_JOBS)

    time_init = time()
    # gFun.fit(lX, ly)

    print('Projecting...')
    y_ = gFun.transform(lX)
    train_time = round(time() - time_init, 3)
    exit(f'Executed! Training time: {train_time}!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()
    main(args)
