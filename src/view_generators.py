"""
This module contains the view generators that take care of computing the view specific document embeddings:

- VanillaFunGen (-x) cast document representations encoded via TFIDF into posterior probabilities by means of SVM.

- WordClassGen (-w): generates document representation via Word-Class-Embeddings.
    Document embeddings are obtained via weighted sum of document's constituent embeddings.

- MuseGen (-m): generates document representation via MUSE embeddings.
    Document embeddings are obtained via weighted sum of document's constituent embeddings.

- RecurrentGen (-g): generates document embedding by means of a Gated Recurrent Units. The model can be
    initialized with different (multilingual/aligned) word representations (e.g., MUSE, WCE, ecc.,).
    Output dimension is (n_docs, 512).

- View generator (-b): generates document embedding via mBERT model.
"""
from abc import ABC, abstractmethod
import time

import torch
import transformers
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from src.data.datamodule import RecurrentDataModule, BertDataModule, tokenize
from src.models.learners import *
from src.models.pl_bert import BertModel
from src.models.pl_gru import RecurrentModel
from src.util.common import TfidfVectorizerMultilingual, _normalize, index
from src.util.embeddings_manager import MuseLoader, XdotM, wce_matrix
from src.util.file import create_if_not_exist
from src.util.csv_logger import CSVLog
from src.models.transformer_nets import *
from src.util.old_earlystop import OldEarlyStopping


transformers.logging.set_verbosity_error()

class ViewGen(ABC):
    """
    Abstract class for ViewGenerators implementations. Every ViewGen should implement these three methods in order to
    be seamlessly integrated in the overall architecture.
    """
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
    """
    View Generator (x): original funnelling architecture proposed by Moreo, Esuli and
    Sebastiani in DOI: https://doi.org/10.1145/3326065
    """
    def __init__(self, base_learner, first_tier_parameters=None, n_jobs=-1):
        """
        Init Posterior Probabilities embedder (i.e., VanillaFunGen)
        :param base_learner: naive monolingual learners to be deployed as first-tier learners. Should be able to
        return posterior probabilities.
        :param base_learner:
        :param n_jobs: integer, number of concurrent workers
        """
        super().__init__()
        self.learners = base_learner
        self.first_tier_parameters = first_tier_parameters
        self.n_jobs = n_jobs
        self.doc_projector = NaivePolylingualClassifier(base_learner=self.learners,
                                                        parameters=self.first_tier_parameters, n_jobs=self.n_jobs)
        self.vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    def fit(self, lX, lY):
        print('# Fitting VanillaFunGen (X)...')
        lX = self.vectorizer.fit_transform(lX)
        self.doc_projector.fit(lX, lY)
        return self

    def transform(self, lX):
        """
        (1) Vectorize documents; (2) Project them according to the learners SVMs, finally (3) Apply L2 normalization
        to the projection and returns it.
        :param lX: dict {lang: indexed documents}
        :return: document projection to the common latent space.
        """
        lX = self.vectorizer.transform(lX)
        lZ = self.doc_projector.predict_proba(lX)
        lZ = _normalize(lZ, l2=True)
        return lZ

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class MuseGen(ViewGen):
    """
    View Generator (m): generates document representation via MUSE embeddings (Fasttext multilingual word
    embeddings). Document embeddings are obtained via weighted sum of document's constituent embeddings.
    """
    def __init__(self, muse_dir='../embeddings', n_jobs=-1):
        """
        Init the MuseGen.
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
        """
        (1) Vectorize documents; (2) Load muse embeddings for words encountered while vectorizing.
        :param lX: dict {lang: indexed documents}
        :param ly: dict {lang: target vectors}
        :return: self.
        """
        print('# Fitting MuseGen (M)...')
        self.vectorizer.fit(lX)
        self.langs = sorted(lX.keys())
        self.lMuse = MuseLoader(langs=self.langs, cache=self.muse_dir)
        lVoc = self.vectorizer.vocabulary()
        self.lMuse = self.lMuse.extract(lVoc)  # overwriting lMuse with dict {lang : embed_matrix} with only known words
        # TODO: featureweight.fit
        return self

    def transform(self, lX):
        """
        (1) Vectorize documents; (2) computes the weighted sum of MUSE embeddings found at document level,
         finally (3) Apply L2 normalization embedding and returns it.
        :param lX: dict {lang: indexed documents}
        :return: document projection to the common latent space.
        """
        lX = self.vectorizer.transform(lX)
        XdotMUSE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], self.lMuse[lang], sif=True) for lang in self.langs)
        lZ = {lang: XdotMUSE[i] for i, lang in enumerate(self.langs)}
        lZ = _normalize(lZ, l2=True)
        return lZ

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class WordClassGen(ViewGen):
    """
    View Generator (w): generates document representation via Word-Class-Embeddings.
    Document embeddings are obtained via weighted sum of document's constituent embeddings.
    """
    def __init__(self, n_jobs=-1):
        """
        Init WordClassGen.
        :param n_jobs: int, number of concurrent workers
        """
        super().__init__()
        self.n_jobs = n_jobs
        self.langs = None
        self.lWce = None
        self.vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    def fit(self, lX, ly):
        """
        (1) Vectorize documents; (2) Load muse embeddings for words encountered while vectorizing.
        :param lX: dict {lang: indexed documents}
        :param ly: dict {lang: target vectors}
        :return: self.
        """
        print('# Fitting WordClassGen (W)...')
        lX = self.vectorizer.fit_transform(lX)
        self.langs = sorted(lX.keys())
        wce = Parallel(n_jobs=self.n_jobs)(
            delayed(wce_matrix)(lX[lang], ly[lang]) for lang in self.langs)
        self.lWce = {l: wce[i] for i, l in enumerate(self.langs)}
        # TODO: featureweight.fit()
        return self

    def transform(self, lX):
        """
        (1) Vectorize documents; (2) computes the weighted sum of Word-Class Embeddings found at document level,
         finally (3) Apply L2 normalization embedding and returns it.
        :param lX: dict {lang: indexed documents}
        :return: document projection to the common latent space.
        """
        lX = self.vectorizer.transform(lX)
        XdotWce = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], self.lWce[lang], sif=True) for lang in self.langs)
        lWce = {l: XdotWce[i] for i, l in enumerate(self.langs)}
        lWce = _normalize(lWce, l2=True)
        return lWce

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class RecurrentGen(ViewGen):
    """
    View Generator (G): generates document embedding by means of a Gated Recurrent Units. The model can be
    initialized with different (multilingual/aligned) word representations (e.g., MUSE, WCE, ecc.,).
    Output dimension is (n_docs, 512). The training will happen end-to-end. At inference time, the model returns
    the network internal state at the second feed-forward layer level. Training metrics are logged via TensorBoard.
    """
    def __init__(self, multilingualIndex, pretrained_embeddings, wce, batch_size=512, nepochs=50,
                 gpus=0, n_jobs=-1, patience=20, stored_path=None):
        """
        Init RecurrentGen.
        :param multilingualIndex: MultilingualIndex, it is a dictionary of training and test documents
        indexed by language code.
        :param pretrained_embeddings: dict {lang: tensor of embeddings}, it contains the pretrained embeddings to use
        as embedding layer.
        :param wce: Bool, whether to deploy Word-Class Embeddings (as proposed by A. Moreo). If True, supervised
        embeddings are concatenated to the deployed supervised embeddings. WCE dimensionality is equal to
        the number of target classes.
        :param batch_size: int, number of samples in a batch.
        :param nepochs: int, number of max epochs to train the model.
        :param gpus: int,  specifies how many GPUs to use per node. If False computation will take place on cpu.
        :param n_jobs: int, number of concurrent workers (i.e., parallelizing data loading).
        :param patience: int, number of epochs with no improvements in val-macroF1 before early stopping.
        :param stored_path: str, path to a pretrained model. If None the model will be trained from scratch.
        """
        super().__init__()
        self.multilingualIndex = multilingualIndex
        self.langs = multilingualIndex.langs
        self.batch_size = batch_size
        self.gpus = gpus
        self.n_jobs = n_jobs
        self.stored_path = stored_path
        self.nepochs = nepochs
        self.patience = patience

        # EMBEDDINGS to be deployed
        self.pretrained = pretrained_embeddings
        self.wce = wce

        self.multilingualIndex.train_val_split(val_prop=0.2, max_val=2000, seed=1)
        self.multilingualIndex.embedding_matrices(self.pretrained, supervised=self.wce)
        self.model = self._init_model()
        self.logger = TensorBoardLogger(save_dir='tb_logs', name='rnn', default_hp_metric=False)
        self.early_stop_callback = EarlyStopping(monitor='val-macroF1', min_delta=0.00,
                                                 patience=self.patience, verbose=False, mode='max')

        # modifying EarlyStopping global var in order to compute >= with respect to the best score
        self.early_stop_callback.mode_dict['max'] = torch.ge

        self.lr_monitor = LearningRateMonitor(logging_interval='epoch')

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
        Train the Neural Network end-to-end.
        lX and ly are not directly used. We rather get them from the multilingual index used in the instantiation
        of the Dataset object (RecurrentDataset) in the GfunDataModule class.
        :param lX: dict {lang: indexed documents}
        :param ly: dict {lang: target vectors}
        :return: self.
        """
        print('# Fitting RecurrentGen (G)...')
        create_if_not_exist(self.logger.save_dir)
        recurrentDataModule = RecurrentDataModule(self.multilingualIndex, batchsize=self.batch_size, n_jobs=self.n_jobs)
        trainer = Trainer(gradient_clip_val=1e-1, gpus=self.gpus, logger=self.logger, max_epochs=self.nepochs,
                          callbacks=[self.early_stop_callback, self.lr_monitor], checkpoint_callback=False)

        trainer.fit(self.model, datamodule=recurrentDataModule)
        trainer.test(self.model, datamodule=recurrentDataModule)
        return self

    def transform(self, lX):
        """
        Project documents to the common latent space. Output dimensionality is 512.
        :param lX: dict {lang: indexed documents}
        :return: documents projected to the common latent space.
        """
        data = {}
        for lang in lX.keys():
            indexed = index(data=lX[lang],
                            vocab=self.multilingualIndex.l_index[lang].word2index,
                            known_words=set(self.multilingualIndex.l_index[lang].word2index.keys()),
                            analyzer=self.multilingualIndex.l_vectorizer.get_analyzer(lang),
                            unk_index=self.multilingualIndex.l_index[lang].unk_index,
                            out_of_vocabulary=self.multilingualIndex.l_index[lang].out_of_vocabulary)
            data[lang] = indexed
        l_pad = self.multilingualIndex.l_pad()
        self.model.to('cuda' if self.gpus else 'cpu')
        self.model.eval()
        l_embeds = self.model.encode(data, l_pad, batch_size=256)
        return l_embeds

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class BertGen(ViewGen):
    """
    View Generator (b):  generates document embedding via Bert model. The training happens end-to-end.
    At inference time, the model returns the network internal state at the last original layer (i.e. 12th). Document
    embeddings are the state associated with the "start" token. Training metrics are logged via TensorBoard.
    """
    def __init__(self, multilingualIndex, batch_size=128, nepochs=50, gpus=0, n_jobs=-1, patience=5, stored_path=None):
        """
        Init Bert model
        :param multilingualIndex: MultilingualIndex, it is a dictionary of training and test documents
        indexed by language code.
        :param batch_size: int, number of samples per batch.
        :param nepochs: int, number of max epochs to train the model.
        :param gpus: int,  specifies how many GPUs to use per node. If False computation will take place on cpu.
        :param patience: int, number of epochs with no improvements in val-macroF1 before early stopping.
        :param n_jobs: int, number of concurrent workers.
        :param stored_path: str, path to a pretrained model. If None the model will be trained from scratch.
        """
        super().__init__()
        self.multilingualIndex = multilingualIndex
        self.nepochs = nepochs
        self.gpus = gpus
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.stored_path = stored_path
        self.model = self._init_model()
        self.patience = patience
        self.logger = TensorBoardLogger(save_dir='tb_logs', name='bert', default_hp_metric=False)
        self.early_stop_callback = EarlyStopping(monitor='val-macroF1', min_delta=0.00,
                                                 patience=self.patience, verbose=False, mode='max')

        # modifying EarlyStopping global var in order to compute >= with respect to the best score
        self.early_stop_callback.mode_dict['max'] = torch.ge

    def _init_model(self):
        output_size = self.multilingualIndex.get_target_dim()
        return BertModel(output_size=output_size, stored_path=self.stored_path, gpus=self.gpus)

    def fit(self, lX, ly):
        """
        Train the Neural Network end-to-end.
        lX and ly are not directly used. We rather get them from the multilingual index used in the instantiation
        of the Dataset object (RecurrentDataset) in the GfunDataModule class.
        :param lX: dict {lang: indexed documents}
        :param ly: dict {lang: target vectors}
        :return: self.
        """
        if self.stored_path is not None:
            print('# SKIPPING Fitting BertGen (B) - we have loaded a trained model!')
            # if we are loading a trained model - avoid training!
            return self
        print('# Fitting BertGen (B)...')
        create_if_not_exist(self.logger.save_dir)
        self.multilingualIndex.train_val_split(val_prop=0.2, max_val=2000, seed=1)
        bertDataModule = BertDataModule(self.multilingualIndex, batchsize=self.batch_size, max_len=512)
        trainer = Trainer(max_epochs=self.nepochs, gpus=self.gpus, logger=self.logger,
                          callbacks=[self.early_stop_callback], checkpoint_callback=False)  # gradient_clip_val=1e-1
        trainer.fit(self.model, datamodule=bertDataModule)
        # trainer.test(self.model, datamodule=bertDataModule)
        self.model.bert.save_pretrained("checkpoints/bert_secondround")
        return self

    def transform(self, lX):
        """
        Project documents to the common latent space. Output dimensionality is 768.
        :param lX: dict {lang: indexed documents}
        :return: documents projected to the common latent space.
        """
        data = tokenize(lX, max_len=512)
        self.model.to('cuda' if self.gpus else 'cpu')
        self.model.eval()
        l_embeds = self.model.encode(data, batch_size=64)
        return l_embeds

    def fit_transform(self, lX, ly):
        # we can assume that we have already indexed data for transform() since we are first calling fit()
        return self.fit(lX, ly).transform(lX)


class OldBertGen(ViewGen):

    def __init__(self,
                 options,
                 doc_embed_path=None,
                 patience=10,
                 checkpoint_dir="checkpoints/bert",
                 path_to_model=None,
                 nC=None):

        self.doc_embed_path = doc_embed_path
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.requires_tfidf = False
        self.options = options
        self.path_to_model = path_to_model
        if self.path_to_model is None:
            print(f'# Init BertModel (B)...')
            self.model = get_model(nC).cuda()
        else:
            print(f'# Initializing own pre-trained BertModel (B)...')
            config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True,
                                                num_labels=nC)
            self.model = BertForSequenceClassification.from_pretrained(path_to_model, config=config).cuda()

    def fit(self, lX, ly, lV=None, seed=0, nepochs=25, lr=1e-5, val_epochs=1):
        if self.path_to_model is not None:
            print('# SKIPPING Fitting BertGen (B) - we have loaded a trained model!')
            # if we are loading a trained model - avoid training!
            return self
        print(f'# Tokenizing data (B)...')
        l_tokenized_tr = do_tokenization(lX, max_len=512)
        l_split_tr, l_split_tr_target, l_split_va, l_split_val_target = get_tr_val_split(l_tokenized_tr,
                                                                                         ly,
                                                                                         val_prop=0.2,
                                                                                         max_val=2000,
                                                                                         seed=seed)

        tr_dataset = TrainingDataset(l_split_tr, l_split_tr_target)
        va_dataset = TrainingDataset(l_split_va, l_split_val_target)
        tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)
        va_dataloader = DataLoader(va_dataset, batch_size=2, shuffle=True)

        print('# Fitting BertGen (B)...')
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
        optim = init_optimizer(self.model, lr=lr, weight_decay=0.01)
        lr_scheduler = StepLR(optim, step_size=25, gamma=0.1)
        early_stop = OldEarlyStopping(self.model,
                                      optimizer=optim,
                                      patience=self.patience,
                                      checkpoint=self.checkpoint_dir,
                                      is_bert=True)

        # Training loop
        logfile_name = 'log_mBert_extractor.csv'
        method_name = 'mBert'
        logfile = new_init_logfile_nn(filename=logfile_name, opt=self.options)

        tinit = time()
        lang_ids = va_dataset.lang_ids
        nepochs = self.options.nepochs_bert
        for epoch in range(1, nepochs + 1):
            print('# Start Training ...')
            train(self.model, tr_dataloader, epoch, criterion, optim, method_name, tinit, logfile, self.options)
            lr_scheduler.step()

            # Validation
            macrof1 = test(self.model, va_dataloader, lang_ids, tinit, epoch, logfile, criterion, 'va')
            early_stop(macrof1, epoch)

            if early_stop.STOP:
                print('[early-stop] STOP')
                break

        model = early_stop.restore_checkpoint()
        self.model = model.cuda()

        if val_epochs > 0:
            print(f'running last {val_epochs} training epochs on the validation set')
            for val_epoch in range(1, val_epochs + 1):
                train(self.model, va_dataloader, epoch + val_epoch, criterion, optim, method_name, tinit, logfile,
                      self.options)

        self.fitted = True
        return self

    def transform(self, lX):
        assert self.fitted is True, 'Calling transform without any initialized model! - call init first or on init' \
                                       'pass the "path_to_model" arg.'
        print('# Obtaining document embeddings from BertGen (B)... ')
        l_tokenized_X = do_tokenization(lX, max_len=512, verbose=True)
        feat_dataset = ExtractorDataset(l_tokenized_X)
        feat_lang_ids = feat_dataset.lang_ids
        dataloader = DataLoader(feat_dataset, batch_size=64)
        all_batch_embeddings, id2lang = feature_extractor(dataloader, feat_lang_ids, self.model)
        return all_batch_embeddings

    def fit_transform(self, lX, ly, lV=None):
        return self.fit(lX, ly).transform(lX)
