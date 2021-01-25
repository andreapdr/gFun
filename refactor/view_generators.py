"""
This module contains the view generators that take care of computing the view specific document embeddings:

- VanillaFunGen (-X) cast document representations encoded via TFIDF into posterior probabilities by means of SVM.

- WordClassGen (-W): generates document representation via Word-Class-Embeddings.
    Document embeddings are obtained via weighted sum of document's constituent embeddings.

- MuseGen (-M):

- RecurrentGen (-G): generates document embedding by means of a Gated Recurrent Units. The model can be
    initialized with different (multilingual/aligned) word representations (e.g., MUSE, WCE, ecc.,).
    Output dimension is (n_docs, 512).

- View generator (-B): generates document embedding via mBERT model.
"""
from abc import ABC, abstractmethod
from models.learners import *
from util.embeddings_manager import MuseLoader, XdotM, wce_matrix
from util.common import TfidfVectorizerMultilingual, _normalize
from models.pl_gru import RecurrentModel
from models.pl_bert import BertModel
from pytorch_lightning import Trainer
from data.datamodule import RecurrentDataModule, BertDataModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from time import time


class ViewGen(ABC):
    @abstractmethod
    def fit(self, lX, ly):
        pass

    @abstractmethod
    def transform(self, lX):
        pass

    @abstractmethod
    def fit_transform(self, lX, ly):
        pass


class VanillaFunGen(ViewGen):
    def __init__(self, base_learner, n_jobs=-1):
        """
        Original funnelling architecture proposed by Moreo, Esuli and Sebastiani in DOI: https://doi.org/10.1145/3326065
        :param base_learner: naive monolingual learners to be deployed as first-tier learners. Should be able to
        return posterior probabilities.
        :param n_jobs: integer, number of concurrent workers
        """
        super().__init__()
        self.learners = base_learner
        self.n_jobs = n_jobs
        self.doc_projector = NaivePolylingualClassifier(self.learners)
        self.vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    def fit(self, lX, lY):
        print('# Fitting VanillaFunGen...')
        lX = self.vectorizer.fit_transform(lX)
        self.doc_projector.fit(lX, lY)
        return self

    def transform(self, lX):
        lX = self.vectorizer.transform(lX)
        lZ = self.doc_projector.predict_proba(lX)
        return lZ

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class MuseGen(ViewGen):
    def __init__(self, muse_dir='../embeddings', n_jobs=-1):
        """
        generates document representation via MUSE embeddings (Fasttext multilingual word
        embeddings). Document embeddings are obtained via weighted sum of document's constituent embeddings.
        :param muse_dir: string, path to folder containing muse embeddings
        :param n_jobs: int, number of concurrent workers
        """
        super().__init__()
        self.muse_dir = muse_dir
        self.n_jobs = n_jobs
        self.langs = None
        self.lMuse = None
        self.vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    def fit(self, lX, ly):
        print('# Fitting MuseGen...')
        self.vectorizer.fit(lX)
        self.langs = sorted(lX.keys())
        self.lMuse = MuseLoader(langs=self.langs, cache=self.muse_dir)
        lVoc = self.vectorizer.vocabulary()
        self.lMuse = self.lMuse.extract(lVoc)  # overwriting lMuse with dict {lang : embed_matrix} with only known words
        # TODO: featureweight.fit
        return self

    def transform(self, lX):
        lX = self.vectorizer.transform(lX)
        XdotMUSE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], self.lMuse[lang], sif=True) for lang in self.langs)
        lZ = {lang: XdotMUSE[i] for i, lang in enumerate(self.langs)}
        lZ = _normalize(lZ, l2=True)
        return lZ

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class WordClassGen(ViewGen):
    def __init__(self, n_jobs=-1):
        """
        generates document representation via Word-Class-Embeddings.
        Document embeddings are obtained via weighted sum of document's constituent embeddings.
        :param n_jobs: int, number of concurrent workers
        """
        super().__init__()
        self.n_jobs = n_jobs
        self.langs = None
        self.lWce = None
        self.vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    def fit(self, lX, ly):
        print('# Fitting WordClassGen...')
        lX = self.vectorizer.fit_transform(lX)
        self.langs = sorted(lX.keys())
        wce = Parallel(n_jobs=self.n_jobs)(
            delayed(wce_matrix)(lX[lang], ly[lang]) for lang in self.langs)
        self.lWce = {l: wce[i] for i, l in enumerate(self.langs)}
        # TODO: featureweight.fit()
        return self

    def transform(self, lX):
        lX = self.vectorizer.transform(lX)
        XdotWce = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], self.lWce[lang], sif=True) for lang in self.langs)
        lWce = {l: XdotWce[i] for i, l in enumerate(self.langs)}
        lWce = _normalize(lWce, l2=True)
        return lWce

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class RecurrentGen(ViewGen):
    # TODO: save model https://forums.pytorchlightning.ai/t/how-to-save-hparams-when-not-provided-as-argument-apparently-assigning-to-hparams-is-not-recomended/339/5
    #  Problem: we are passing lPretrained to init the RecurrentModel -> incredible slow at saving (checkpoint).
    #  if we do not save it is impossible to init RecurrentModel by calling RecurrentModel.load_from_checkpoint()
    def __init__(self, multilingualIndex, pretrained_embeddings, wce, batch_size=512, nepochs=50,
                 gpus=0, n_jobs=-1, stored_path=None):
        """
        generates document embedding by means of a Gated Recurrent Units. The model can be
        initialized with different (multilingual/aligned) word representations (e.g., MUSE, WCE, ecc.,).
        Output dimension is (n_docs, 512).
        :param multilingualIndex:
        :param pretrained_embeddings:
        :param wce:
        :param gpus:
        :param n_jobs:
        """
        super().__init__()
        self.multilingualIndex = multilingualIndex
        self.langs = multilingualIndex.langs
        self.batch_size = batch_size
        self.gpus = gpus
        self.n_jobs = n_jobs
        self.stored_path = stored_path
        self.nepochs = nepochs

        # EMBEDDINGS to be deployed
        self.pretrained = pretrained_embeddings
        self.wce = wce

        self.multilingualIndex.train_val_split(val_prop=0.2, max_val=2000, seed=1)
        self.multilingualIndex.embedding_matrices(self.pretrained, supervised=self.wce)
        self.model = self._init_model()
        self.logger = TensorBoardLogger(save_dir='tb_logs', name='rnn', default_hp_metric=False)
        # self.logger = CSVLogger(save_dir='csv_logs', name='rnn_dev')

    def _init_model(self):
        if self.stored_path:
            lpretrained = self.multilingualIndex.l_embeddings()
            return RecurrentModel.load_from_checkpoint(self.stored_path, lPretrained=lpretrained)
        else:
            lpretrained = self.multilingualIndex.l_embeddings()
            langs = self.multilingualIndex.langs
            output_size = self.multilingualIndex.get_target_dim()
            hidden_size = 512
            lvocab_size = self.multilingualIndex.l_vocabsize()
            learnable_length = 0
            return RecurrentModel(
                lPretrained=lpretrained,
                langs=langs,
                output_size=output_size,
                hidden_size=hidden_size,
                lVocab_size=lvocab_size,
                learnable_length=learnable_length,
                drop_embedding_range=self.multilingualIndex.sup_range,
                drop_embedding_prop=0.5,
                gpus=self.gpus
            )

    def fit(self, lX, ly):
        """
        lX and ly are not directly used. We rather get them from the multilingual index used in the instantiation
        of the Dataset object (RecurrentDataset) in the GfunDataModule class.
        :param lX:
        :param ly:
        :return:
        """
        print('# Fitting RecurrentGen...')
        recurrentDataModule = RecurrentDataModule(self.multilingualIndex, batchsize=self.batch_size)
        trainer = Trainer(gradient_clip_val=1e-1, gpus=self.gpus, logger=self.logger, max_epochs=self.nepochs,
                          checkpoint_callback=False)

        # vanilla_torch_model = torch.load(
        #     '/home/andreapdr/funneling_pdr/checkpoint/gru_viewgen_-jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run0.pickle')
        # self.model.linear0 = vanilla_torch_model.linear0
        # self.model.linear1 = vanilla_torch_model.linear1
        # self.model.linear2 = vanilla_torch_model.linear2
        # self.model.rnn = vanilla_torch_model.rnn

        trainer.fit(self.model, datamodule=recurrentDataModule)
        trainer.test(self.model, datamodule=recurrentDataModule)
        return self

    def transform(self, lX):
        """
        Project documents to the common latent space
        :param lX:
        :return:
        """
        l_pad = self.multilingualIndex.l_pad()
        data = self.multilingualIndex.l_devel_index()
        self.model.to('cuda' if self.gpus else 'cpu')
        self.model.eval()
        time_init = time()
        l_embeds = self.model.encode(data, l_pad, batch_size=256)
        transform_time = round(time() - time_init, 3)
        print(f'Executed! Transform took: {transform_time}')
        return l_embeds

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class BertGen(ViewGen):
    def __init__(self, multilingualIndex, batch_size=128, nepochs=50, gpus=0, n_jobs=-1, stored_path=None):
        super().__init__()
        self.multilingualIndex = multilingualIndex
        self.nepochs = nepochs
        self.gpus = gpus
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.stored_path = stored_path
        self.model = self._init_model()
        self.logger = TensorBoardLogger(save_dir='tb_logs', name='bert', default_hp_metric=False)

    def _init_model(self):
        output_size = self.multilingualIndex.get_target_dim()
        return BertModel(output_size=output_size, stored_path=self.stored_path, gpus=self.gpus)

    def fit(self, lX, ly):
        print('# Fitting BertGen...')
        self.multilingualIndex.train_val_split(val_prop=0.2, max_val=2000, seed=1)
        bertDataModule = BertDataModule(self.multilingualIndex, batchsize=self.batch_size, max_len=512)
        trainer = Trainer(gradient_clip_val=1e-1, max_epochs=self.nepochs, gpus=self.gpus,
                          logger=self.logger, checkpoint_callback=False)
        trainer.fit(self.model, datamodule=bertDataModule)
        trainer.test(self.model, datamodule=bertDataModule)
        return self

    def transform(self, lX):
        # lX is raw text data. It has to be first indexed via Bert Tokenizer.
        data = 'TOKENIZE THIS'
        self.model.to('cuda' if self.gpus else 'cpu')
        self.model.eval()
        time_init = time()
        l_emebds = self.model.encode(data)  # TODO
        transform_time = round(time() - time_init, 3)
        print(f'Executed! Transform took: {transform_time}')
        exit('BERT VIEWGEN TRANSFORM NOT IMPLEMENTED!')
        return l_emebds

    def fit_transform(self, lX, ly):
        # we can assume that we have already indexed data for transform() since we are first calling fit()
        return self.fit(lX, ly).transform(lX)


