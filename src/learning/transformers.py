from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from data.tsr_function__ import get_tsr_matrix, get_supervised_matrix, pointwise_mutual_information, information_gain
from embeddings.embeddings import FastTextMUSE
from embeddings.supervised import supervised_embeddings_tfidf, zscores
from learning.learners import NaivePolylingualClassifier, MonolingualClassifier, _joblib_transform_multiling
from sklearn.decomposition import PCA
from scipy.sparse import hstack
from util_transformers.StandardizeTransformer import StandardizeTransformer
from util.SIF_embed import remove_pc
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from models.mBert import *
from models.lstm_class import *
from util.csv_log import CSVLog
from util.file import get_file_name
from util.early_stop import EarlyStopping
from util.common import *
import time


# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------


class FeatureWeight:

    def __init__(self, weight='tfidf', agg='mean'):
        assert weight in ['tfidf', 'pmi', 'ig'] or callable(
            weight), 'weight should either be "tfidf" or a callable function'
        assert agg in ['mean', 'max'], 'aggregation function should either be "mean" or "max"'
        self.weight = weight
        self.agg = agg
        self.fitted = False
        if weight == 'pmi':
            self.weight = pointwise_mutual_information
        elif weight == 'ig':
            self.weight = information_gain

    def fit(self, lX, ly):
        if not self.fitted:
            if self.weight == 'tfidf':
                self.lF = {l: np.ones(X.shape[1]) for l, X in lX.items()}
            else:
                self.lF = {}
                for l in lX.keys():
                    X, y = lX[l], ly[l]

                    print(f'getting supervised cell-matrix lang {l}')
                    tsr_matrix = get_tsr_matrix(get_supervised_matrix(X, y), tsr_score_funtion=self.weight)
                    if self.agg == 'max':
                        F = tsr_matrix.max(axis=0)
                    elif self.agg == 'mean':
                        F = tsr_matrix.mean(axis=0)
                    self.lF[l] = F

            self.fitted = True
        return self

    def transform(self, lX):
        return {lang: csr_matrix.multiply(lX[lang], self.lF[lang]) for lang in lX.keys()}

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)

# ------------------------------------------------------------------
# View Generators (aka first-tier learners)
# ------------------------------------------------------------------


class PosteriorProbabilitiesEmbedder:

    def __init__(self, first_tier_learner, first_tier_parameters=None, l2=True, n_jobs=-1):
        self.fist_tier_learner = first_tier_learner
        self.fist_tier_parameters = first_tier_parameters
        self.l2 = l2
        self.n_jobs = n_jobs
        self.doc_projector = NaivePolylingualClassifier(
            self.fist_tier_learner, self.fist_tier_parameters, n_jobs=n_jobs
        )
        self.requires_tfidf = True

    def fit(self, lX, lY, lV=None, called_by_viewgen=False):
        if not called_by_viewgen:
            # Avoid printing if method is called by another View Gen (e.g., GRU ViewGen)
            print('### Posterior Probabilities View Generator (X)')
            print('fitting the projectors... {}'.format(lX.keys()))
        self.doc_projector.fit(lX, lY)
        return self

    def transform(self, lX):
        lZ = self.predict_proba(lX)
        lZ = _normalize(lZ, self.l2)
        return lZ

    def fit_transform(self, lX, ly=None, lV=None):
        return self.fit(lX, ly).transform(lX)

    def best_params(self):
        return self.doc_projector.best_params()

    def predict(self, lX, ly=None):
        return self.doc_projector.predict(lX)

    def predict_proba(self, lX, ly=None):
        print(f'generating posterior probabilities for {sum([X.shape[0] for X in lX.values()])} documents')
        return self.doc_projector.predict_proba(lX)

    def _get_output_dim(self):
        return len(self.doc_projector.model['da'].model.classes_)


class MuseEmbedder:

    def __init__(self, path, lV=None, l2=True, n_jobs=-1, featureweight=FeatureWeight(), sif=False):
        self.path = path
        self.lV = lV
        self.l2 = l2
        self.n_jobs = n_jobs
        self.featureweight = featureweight
        self.sif = sif
        self.requires_tfidf = True

    def fit(self, lX, ly, lV=None):
        assert lV is not None or self.lV is not None, 'lV not specified'
        print('### MUSE View Generator (M)')
        print(f'Loading fastText pretrained vectors for languages {list(lX.keys())}...')
        self.langs = sorted(lX.keys())
        self.MUSE = load_muse_embeddings(self.path, self.langs, self.n_jobs)
        lWordList = {l: self._get_wordlist_from_word2index(lV[l]) for l in self.langs}
        self.MUSE = {l: Muse.extract(lWordList[l]).numpy() for l, Muse in self.MUSE.items()}
        self.featureweight.fit(lX, ly)
        return self

    def transform(self, lX):
        MUSE = self.MUSE
        lX = self.featureweight.transform(lX)
        XdotMUSE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], MUSE[lang], self.sif) for lang in self.langs
        )
        lMuse = {l: XdotMUSE[i] for i, l in enumerate(self.langs)}
        lMuse = _normalize(lMuse, self.l2)
        return lMuse

    def fit_transform(self, lX, ly, lV):
        return self.fit(lX, ly, lV).transform(lX)

    def _get_wordlist_from_word2index(self, word2index):
        return list(zip(*sorted(word2index.items(), key=lambda x: x[1])))[0]

    def _get_output_dim(self):
        return self.MUSE['da'].shape[1]


class WordClassEmbedder:

    def __init__(self, l2=True, n_jobs=-1, max_label_space=300, featureweight=FeatureWeight(), sif=False):
        self.n_jobs = n_jobs
        self.l2 = l2
        self.max_label_space = max_label_space
        self.featureweight = featureweight
        self.sif = sif
        self.requires_tfidf = True

    def fit(self, lX, ly, lV=None):
        print('### WCE View Generator (M)')
        print('Computing supervised embeddings...')
        self.langs = sorted(lX.keys())
        WCE = Parallel(n_jobs=self.n_jobs)(
            delayed(word_class_embedding_matrix)(lX[lang], ly[lang], self.max_label_space) for lang in self.langs
        )
        self.lWCE = {l: WCE[i] for i, l in enumerate(self.langs)}
        self.featureweight.fit(lX, ly)
        return self

    def transform(self, lX):
        lWCE = self.lWCE
        lX = self.featureweight.transform(lX)
        XdotWCE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], lWCE[lang], self.sif) for lang in self.langs
        )
        lwce = {l: XdotWCE[i] for i, l in enumerate(self.langs)}
        lwce = _normalize(lwce, self.l2)
        return lwce

    def fit_transform(self, lX, ly, lV=None):
        return self.fit(lX, ly).transform(lX)

    def _get_output_dim(self):
        return 73   # TODO !


class MBertEmbedder:

    def __init__(self, doc_embed_path=None, patience=10, checkpoint_dir='../hug_checkpoint/', path_to_model=None,
                 nC=None):
        self.doc_embed_path = doc_embed_path
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.fitted = False
        self.requires_tfidf = False
        if path_to_model is None and nC is not None:
            self.model = None
        else:
            config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True,
                                                num_labels=nC)
            self.model = BertForSequenceClassification.from_pretrained(path_to_model, config=config).cuda()
            self.fitted = True

    def fit(self, lX, ly, lV=None, seed=0, nepochs=200, lr=1e-5, val_epochs=1):
        print('### mBERT View Generator (B)')
        if self.fitted is True:
            print('Bert model already fitted!')
            return self

        print('Fine-tune mBert on the given dataset.')
        l_tokenized_tr = do_tokenization(lX, max_len=512)
        l_split_tr, l_split_tr_target, l_split_va, l_split_val_target = get_tr_val_split(l_tokenized_tr, ly,
                                                                                         val_prop=0.2, max_val=2000,
                                                                                         seed=seed)     # TODO: seed

        tr_dataset = TrainingDataset(l_split_tr, l_split_tr_target)
        va_dataset = TrainingDataset(l_split_va, l_split_val_target)
        tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)
        va_dataloader = DataLoader(va_dataset, batch_size=2, shuffle=True)

        nC = tr_dataset.get_nclasses()
        model = get_model(nC)
        model = model.cuda()
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
        optim = init_optimizer(model, lr=lr, weight_decay=0.01)
        lr_scheduler = StepLR(optim, step_size=25, gamma=0.1)
        early_stop = EarlyStopping(model, optimizer=optim, patience=self.patience,
                                   checkpoint=self.checkpoint_dir,
                                   is_bert=True)

        # Training loop
        logfile = '../log/log_mBert_extractor.csv'
        method_name = 'mBert_feature_extractor'

        tinit = time()
        lang_ids = va_dataset.lang_ids
        for epoch in range(1, nepochs + 1):
            print('# Start Training ...')
            train(model, tr_dataloader, epoch, criterion, optim, method_name, tinit, logfile)
            lr_scheduler.step()  # reduces the learning rate # TODO arg epoch?

            # Validation
            macrof1 = test(model, va_dataloader, lang_ids, tinit, epoch, logfile, criterion, 'va')
            early_stop(macrof1, epoch)

            if early_stop.STOP:
                print('[early-stop] STOP')
                break

        model = early_stop.restore_checkpoint()
        self.model = model.cuda()

        if val_epochs > 0:
            print(f'running last {val_epochs} training epochs on the validation set')
            for val_epoch in range(1, val_epochs + 1):
                train(self.model, va_dataloader, epoch + val_epoch, criterion, optim, method_name, tinit, logfile)

        self.fitted = True
        return self

    def transform(self, lX):
        assert self.fitted is True, 'Calling transform without any initialized model! - call init first or on init' \
                                       'pass the "path_to_model" arg.'
        print('Obtaining document embeddings from pretrained mBert ')
        l_tokenized_X = do_tokenization(lX, max_len=512, verbose=True)
        feat_dataset = ExtractorDataset(l_tokenized_X)
        feat_lang_ids = feat_dataset.lang_ids
        dataloader = DataLoader(feat_dataset, batch_size=64)
        all_batch_embeddings, id2lang = feature_extractor(dataloader, feat_lang_ids, self.model)
        return all_batch_embeddings

    def fit_transform(self, lX, ly, lV=None):
        return self.fit(lX, ly).transform(lX)


class RecurrentEmbedder:

    def __init__(self, pretrained, supervised, multilingual_dataset, options, concat=False, lr=1e-3,
                 we_path='../embeddings', hidden_size=512, sup_drop=0.5, posteriors=False, patience=10,
                 test_each=0, checkpoint_dir='../checkpoint', model_path=None):
        self.pretrained = pretrained
        self.supervised = supervised
        self.concat = concat
        self.requires_tfidf = False
        self.multilingual_dataset = multilingual_dataset
        self.model = None
        self.we_path = we_path
        self.langs = multilingual_dataset.langs()
        self.hidden_size = hidden_size
        self.sup_drop = sup_drop
        self.posteriors = posteriors
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.test_each = test_each
        self.options = options
        self.seed = options.seed
        self.is_trained = False

        ## INIT MODEL for training
        self.lXtr, self.lytr = self.multilingual_dataset.training(target_as_csr=True)
        self.lXte, self.lyte = self.multilingual_dataset.test(target_as_csr=True)
        self.nC = self.lyte[self.langs[0]].shape[1]
        lpretrained, lpretrained_vocabulary = self._load_pretrained_embeddings(self.we_path, self.langs)
        self.multilingual_index = MultilingualIndex()
        self.multilingual_index.index(self.lXtr, self.lytr, self.lXte, lpretrained_vocabulary)
        self.multilingual_index.train_val_split(val_prop=0.2, max_val=2000, seed=self.seed)
        self.multilingual_index.embedding_matrices(lpretrained, self.supervised)

        if model_path is not None:
            self.is_trained = True
            self.model = torch.load(model_path)
        else:
            self.model = self._init_Net()

        self.optim = init_optimizer(self.model, lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss().cuda()
        self.lr_scheduler = StepLR(self.optim, step_size=25, gamma=0.5)
        self.early_stop = EarlyStopping(self.model, optimizer=self.optim, patience=self.patience,
                                        checkpoint=f'{self.checkpoint_dir}/gru_viewgen_-{get_file_name(self.options.dataset)}')
        # Init SVM in order to recast (vstacked) document embeddings to vectors of Posterior Probabilities
        self.posteriorEmbedder = MetaClassifier(
            SVC(kernel='rbf', gamma='auto', probability=True, cache_size=1000, random_state=1), n_jobs=options.n_jobs)

    def fit(self, lX, ly, lV=None, batch_size=64, nepochs=200, val_epochs=1):
        print('### Gated Recurrent Unit View Generator (G)')
        if not self.is_trained:
            # Batchify input
            self.multilingual_index.train_val_split(val_prop=0.2, max_val=2000, seed=self.seed)
            l_train_index, l_train_target = self.multilingual_index.l_train()
            l_val_index, l_val_target = self.multilingual_index.l_val()
            l_test_index = self.multilingual_index.l_test_index()
            batcher_train = BatchGRU(batch_size, batches_per_epoch=batch_size, languages=self.langs,
                                     lpad=self.multilingual_index.l_pad())
            batcher_eval = BatchGRU(batch_size, batches_per_epoch=batch_size, languages=self.langs,
                                    lpad=self.multilingual_index.l_pad())

            # Train loop
            print('Start training')
            method_name = 'gru_view_generator'
            logfile = init_logfile_nn(method_name, self.options)
            tinit = time.time()
            for epoch in range(1, nepochs + 1):
                train_gru(model=self.model, batcher=batcher_train, ltrain_index=l_train_index, lytr=l_train_target,
                          tinit=tinit, logfile=logfile, criterion=self.criterion, optim=self.optim,
                          epoch=epoch, method_name=method_name, opt=self.options, ltrain_posteriors=None,
                          ltrain_bert=None)
                self.lr_scheduler.step()  # reduces the learning rate # TODO arg epoch?

                # validation step
                macrof1 = test_gru(self.model, batcher_eval, l_val_index, None, None, l_val_target, tinit, epoch,
                                   logfile, self.criterion, 'va')

                self.early_stop(macrof1, epoch)
                if self.test_each > 0:
                    test_gru(self.model, batcher_eval, l_test_index, None, None, self.lyte, tinit, epoch,
                             logfile, self.criterion, 'te')

                if self.early_stop.STOP:
                    print('[early-stop] STOP')
                    print('Restoring best model...')
                    break

            self.model = self.early_stop.restore_checkpoint()
            print(f'running last {val_epochs} training epochs on the validation set')
            for val_epoch in range(1, val_epochs+1):
                batcher_train.init_offset()
                train_gru(model=self.model, batcher=batcher_train, ltrain_index=l_train_index, lytr=l_train_target,
                          tinit=tinit, logfile=logfile, criterion=self.criterion, optim=self.optim,
                          epoch=epoch, method_name=method_name, opt=self.options, ltrain_posteriors=None,
                          ltrain_bert=None)
            self.is_trained = True

        # Generate document embeddings in order to fit an SVM to recast them as vector for Posterior Probabilities
        lX = self._get_doc_embeddings(lX)
        # Fit a ''multi-lingual'' SVM on the generated doc embeddings
        self.posteriorEmbedder.fit(lX, ly)
        return self

    def transform(self, lX, batch_size=64):
        lX = self._get_doc_embeddings(lX)
        return self.posteriorEmbedder.predict_proba(lX)

    def fit_transform(self, lX, ly, lV=None):
        # TODO
        return 0

    def _get_doc_embeddings(self, lX, batch_size=64):
        assert self.is_trained, 'Model is not trained, cannot call transform before fitting the model!'
        print('Generating document embeddings via GRU')
        lX = {}
        ly = {}
        batcher_transform = BatchGRU(batch_size, batches_per_epoch=batch_size, languages=self.langs,
                                     lpad=self.multilingual_index.l_pad())

        l_devel_index = self.multilingual_index.l_devel_index()
        l_devel_target = self.multilingual_index.l_devel_target()

        for idx, (batch, post, bert_emb, target, lang) in enumerate(
                batcher_transform.batchify(l_devel_index, None, None, l_devel_target)):
            if lang not in lX.keys():
                lX[lang] = self.model.get_embeddings(batch, lang)
                ly[lang] = target.cpu().detach().numpy()
            else:
                lX[lang] = np.concatenate((lX[lang], self.model.get_embeddings(batch, lang)), axis=0)
                ly[lang] = np.concatenate((ly[lang], target.cpu().detach().numpy()), axis=0)

        return lX

    # loads the MUSE embeddings if requested, or returns empty dictionaries otherwise
    def _load_pretrained_embeddings(self, we_path, langs):
        lpretrained = lpretrained_vocabulary = self._none_dict(langs) # TODO ?
        lpretrained = load_muse_embeddings(we_path, langs, n_jobs=-1)
        lpretrained_vocabulary = {l: lpretrained[l].vocabulary() for l in langs}
        return lpretrained, lpretrained_vocabulary

    def _none_dict(self, langs):
        return {l:None for l in langs}

    # instantiates the net, initializes the model parameters, and sets embeddings trainable if requested
    def _init_Net(self, xavier_uniform=True):
        model = RNNMultilingualClassifier(
            output_size=self.nC,
            hidden_size=self.hidden_size,
            lvocab_size=self.multilingual_index.l_vocabsize(),
            learnable_length=0,
            lpretrained=self.multilingual_index.l_embeddings(),
            drop_embedding_range=self.multilingual_index.sup_range,
            drop_embedding_prop=self.sup_drop,
            post_probabilities=self.posteriors
        )
        return model.cuda()


class DocEmbedderList:

    def __init__(self, *embedder_list, aggregation='concat'):
        assert aggregation in {'concat', 'mean'}, 'unknown aggregation mode, valid are "concat" and "mean"'
        if len(embedder_list) == 0:
            embedder_list = []
        self.embedders = embedder_list
        self.aggregation = aggregation
        print(f'Aggregation mode: {self.aggregation}')

    def fit(self, lX, ly, lV=None, tfidf=None):
        for transformer in self.embedders:
            _lX = lX
            if transformer.requires_tfidf:
                _lX = tfidf
            transformer.fit(_lX, ly, lV)
        return self

    def transform(self, lX, tfidf=None):
        if self.aggregation == 'concat':
            return self.transform_concat(lX, tfidf)
        elif self.aggregation == 'mean':
            return self.transform_mean(lX, tfidf)

    def transform_concat(self, lX, tfidf):
        if len(self.embedders) == 1:
            if self.embedders[0].requires_tfidf:
                lX = tfidf
            return self.embedders[0].transform(lX)

        some_sparse = False
        langs = sorted(lX.keys())

        lZparts = {l: [] for l in langs}
        for transformer in self.embedders:
            _lX = lX
            if transformer.requires_tfidf:
                _lX = tfidf
            lZ = transformer.transform(_lX)
            for l in langs:
                Z = lZ[l]
                some_sparse = some_sparse or issparse(Z)
                lZparts[l].append(Z)

        hstacker = hstack if some_sparse else np.hstack
        return {l: hstacker(lZparts[l]) for l in langs}

    def transform_mean(self, lX, tfidf):
        if len(self.embedders) == 1:
            return self.embedders[0].transform(lX)

        langs = sorted(lX.keys())

        lZparts = {l: None for l in langs}

        # min_dim = min([transformer._get_output_dim() for transformer in self.embedders])
        min_dim = 73        # TODO <---- this should be the number of target classes

        for transformer in self.embedders:
            _lX = lX
            if transformer.requires_tfidf:
                _lX = tfidf
            lZ = transformer.transform(_lX)
            nC = min([lZ[lang].shape[1] for lang in langs])
            for l in langs:
                Z = lZ[l]
                if Z.shape[1] > min_dim:
                    print(
                        f'Space Z matrix has more dimensions ({Z.shape[1]}) than the smallest representation {min_dim}.'
                        f'Applying PCA(n_components={min_dim})')
                    pca = PCA(n_components=min_dim)
                    Z = pca.fit(Z).transform(Z)
                if lZparts[l] is None:
                    lZparts[l] = Z
                else:
                    lZparts[l] += Z

        n_transformers = len(self.embedders)

        return {l: lZparts[l] / n_transformers for l in langs}

    def fit_transform(self, lX, ly, lV=None, tfidf=None):
        return self.fit(lX, ly, lV, tfidf).transform(lX, tfidf)

    def best_params(self):
        return {'todo'}

    def append(self, embedder):
        self.embedders.append(embedder)


class FeatureSet2Posteriors:
    def __init__(self, transformer, requires_tfidf=False, l2=True, n_jobs=-1):
        self.transformer = transformer
        self.l2 = l2
        self.n_jobs = n_jobs
        self.prob_classifier = MetaClassifier(
            SVC(kernel='rbf', gamma='auto', probability=True, cache_size=1000, random_state=1), n_jobs=n_jobs)
        self.requires_tfidf = requires_tfidf

    def fit(self, lX, ly, lV=None):
        if lV is None and hasattr(self.transformer, 'lV'):
            lV = self.transformer.lV
        lZ = self.transformer.fit_transform(lX, ly, lV)
        self.prob_classifier.fit(lZ, ly)
        return self

    def transform(self, lX):
        lP = self.predict_proba(lX)
        lP = _normalize(lP, self.l2)
        return lP

    def fit_transform(self, lX, ly, lV):
        return self.fit(lX, ly, lV).transform(lX)

    def predict(self, lX, ly=None):
        lZ = self.transformer.transform(lX)
        return self.prob_classifier.predict(lZ)

    def predict_proba(self, lX, ly=None):
        lZ = self.transformer.transform(lX)
        return self.prob_classifier.predict_proba(lZ)


# ------------------------------------------------------------------
# Meta-Classifier (aka second-tier learner)
# ------------------------------------------------------------------
class MetaClassifier:

    def __init__(self, meta_learner, meta_parameters=None, n_jobs=-1, standardize_range=None):
        self.n_jobs = n_jobs
        self.model = MonolingualClassifier(base_learner=meta_learner, parameters=meta_parameters, n_jobs=n_jobs)
        self.standardize_range = standardize_range

    def fit(self, lZ, ly):
        tinit = time.time()
        Z, y = self.stack(lZ, ly)

        self.standardizer = StandardizeTransformer(range=self.standardize_range)
        Z = self.standardizer.fit_transform(Z)

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model.fit(Z, y)
        self.time = time.time() - tinit

    def stack(self, lZ, ly=None):
        langs = list(lZ.keys())
        Z = np.vstack([lZ[lang] for lang in langs])  # Z is the language independent space
        if ly is not None:
            y = np.vstack([ly[lang] for lang in langs])
            return Z, y
        else:
            return Z

    def predict(self, lZ, ly=None):
        lZ = _joblib_transform_multiling(self.standardizer.transform, lZ, n_jobs=self.n_jobs)
        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)

    def predict_proba(self, lZ, ly=None):
        lZ = _joblib_transform_multiling(self.standardizer.transform, lZ, n_jobs=self.n_jobs)
        return _joblib_transform_multiling(self.model.predict_proba, lZ, n_jobs=self.n_jobs)

    def best_params(self):
        return self.model.best_params()


# ------------------------------------------------------------------
# Ensembling (aka Funnelling)
# ------------------------------------------------------------------
class Funnelling:
    def __init__(self,
                 vectorizer: TfidfVectorizerMultilingual,
                 first_tier: DocEmbedderList,
                 meta: MetaClassifier):
        self.vectorizer = vectorizer
        self.first_tier = first_tier
        self.meta = meta
        self.n_jobs = meta.n_jobs

    def fit(self, lX, ly):
        tfidf_lX = self.vectorizer.fit_transform(lX, ly)
        lV = self.vectorizer.vocabulary()
        print('## Fitting first-tier learners!')
        lZ = self.first_tier.fit_transform(lX, ly, lV, tfidf=tfidf_lX)
        print('## Fitting meta-learner!')
        self.meta.fit(lZ, ly)

    def predict(self, lX, ly=None):
        tfidf_lX = self.vectorizer.transform(lX)
        lZ = self.first_tier.transform(lX, tfidf=tfidf_lX)
        ly_ = self.meta.predict(lZ)
        return ly_

    def best_params(self):
        return {'1st-tier': self.first_tier.best_params(),
                'meta': self.meta.best_params()}


class Voting:
    def __init__(self, *prob_classifiers):
        assert all([hasattr(p, 'predict_proba') for p in prob_classifiers]), 'not all classifiers are probabilistic'
        self.prob_classifiers = prob_classifiers

    def fit(self, lX, ly, lV=None):
        for classifier in self.prob_classifiers:
            classifier.fit(lX, ly, lV)

    def predict(self, lX, ly=None):
        lP = {l: [] for l in lX.keys()}
        for classifier in self.prob_classifiers:
            lPi = classifier.predict_proba(lX)
            for l in lX.keys():
                lP[l].append(lPi[l])

        lP = {l: np.stack(Plist).mean(axis=0) for l, Plist in lP.items()}
        ly = {l: P > 0.5 for l, P in lP.items()}

        return ly


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def load_muse_embeddings(we_path, langs, n_jobs=-1):
    MUSE = Parallel(n_jobs=n_jobs)(
        delayed(FastTextMUSE)(we_path, lang) for lang in langs
    )
    return {l: MUSE[i] for i, l in enumerate(langs)}


def word_class_embedding_matrix(X, Y, max_label_space=300):
    WCE = supervised_embeddings_tfidf(X, Y)
    WCE = zscores(WCE, axis=0)

    nC = Y.shape[1]
    if nC > max_label_space:
        print(f'supervised matrix has more dimensions ({nC}) than the allowed limit {max_label_space}. '
              f'Applying PCA(n_components={max_label_space})')
        pca = PCA(n_components=max_label_space)
        WCE = pca.fit(WCE).transform(WCE)

    return WCE


def XdotM(X, M, sif):
    E = X.dot(M)
    if sif:
        print("removing pc...")
        E = remove_pc(E, npc=1)
    return E


def _normalize(lX, l2=True):
    return {l: normalize(X) for l, X in lX.items()} if l2 else lX


class BatchGRU:
    def __init__(self, batchsize, batches_per_epoch, languages, lpad, max_pad_length=500):
        self.batchsize = batchsize
        self.batches_per_epoch = batches_per_epoch
        self.languages = languages
        self.lpad=lpad
        self.max_pad_length=max_pad_length
        self.init_offset()

    def init_offset(self):
        self.offset = {lang: 0 for lang in self.languages}

    def batchify(self, l_index, l_post, l_bert, llabels):
        langs = self.languages
        l_num_samples = {l:len(l_index[l]) for l in langs}

        max_samples = max(l_num_samples.values())
        n_batches = max_samples // self.batchsize + 1 * (max_samples % self.batchsize > 0)
        if self.batches_per_epoch != -1 and self.batches_per_epoch < n_batches:
            n_batches = self.batches_per_epoch

        for b in range(n_batches):
            for lang in langs:
                index, labels = l_index[lang], llabels[lang]
                offset = self.offset[lang]
                if offset >= l_num_samples[lang]:
                    offset = 0
                limit = offset+self.batchsize

                batch_slice = slice(offset, limit)
                batch = index[batch_slice]
                batch_labels = labels[batch_slice].toarray()

                post = None
                bert_emb = None

                batch = pad(batch, pad_index=self.lpad[lang], max_pad_length=self.max_pad_length)
                batch = torch.LongTensor(batch).cuda()
                target = torch.FloatTensor(batch_labels).cuda()

                self.offset[lang] = limit

                yield batch, post, bert_emb, target, lang


def pad(index_list, pad_index, max_pad_length=None):
    pad_length = np.max([len(index) for index in index_list])
    if max_pad_length is not None:
        pad_length = min(pad_length, max_pad_length)
    for i,indexes in enumerate(index_list):
        index_list[i] = [pad_index]*(pad_length-len(indexes)) + indexes[:pad_length]
    return index_list


def train_gru(model, batcher, ltrain_index, lytr, tinit, logfile, criterion, optim, epoch, method_name, opt,
              ltrain_posteriors=None, ltrain_bert=None, log_interval=10):
    _dataset_path = opt.dataset.split('/')[-1].split('_')
    dataset_id = _dataset_path[0] + _dataset_path[-1]

    loss_history = []
    model.train()
    for idx, (batch, post, bert_emb, target, lang) in enumerate(batcher.batchify(ltrain_index, ltrain_posteriors, ltrain_bert, lytr)):
        optim.zero_grad()
        loss = criterion(model(batch, post, bert_emb, lang), target)
        loss.backward()
        clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % log_interval == 0:
            interval_loss = np.mean(loss_history[-log_interval:])
            print(f'{dataset_id} {method_name} Epoch: {epoch}, Step: {idx}, lr={get_lr(optim):.5f}, '
                  f'Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    logfile.add_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time.time() - tinit)
    return mean_loss


def test_gru(model, batcher, ltest_index, ltest_posteriors, lte_bert, lyte, tinit, epoch, logfile, criterion, measure_prefix):
    loss_history = []
    model.eval()
    langs = sorted(ltest_index.keys())
    predictions = {l: [] for l in langs}
    yte_stacked = {l: [] for l in langs}
    batcher.init_offset()
    for batch, post, bert_emb, target, lang in tqdm(batcher.batchify(ltest_index, ltest_posteriors, lte_bert, lyte),
                                                    desc='evaluation: '):
        logits = model(batch, post, bert_emb, lang)
        loss = criterion(logits, target).item()
        prediction = predict(logits)
        predictions[lang].append(prediction)
        yte_stacked[lang].append(target.detach().cpu().numpy())
        loss_history.append(loss)

    ly  = {l:np.vstack(yte_stacked[l]) for l in langs}
    ly_ = {l:np.vstack(predictions[l]) for l in langs}
    l_eval = evaluate(ly, ly_)
    metrics = []
    for lang in langs:
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        if measure_prefix == 'te':
            print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
    Mf1, mF1, MK, mk = np.mean(np.array(metrics), axis=0)
    print(f'[{measure_prefix}] Averages: MF1, mF1, MK, mK [{Mf1:.5f}, {mF1:.5f}, {MK:.5f}, {mk:.5f}]')

    mean_loss = np.mean(loss_history)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=time.time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mF1, timelapse=time.time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-K', value=MK, timelapse=time.time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-K', value=mk, timelapse=time.time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=mean_loss, timelapse=time.time() - tinit)

    return Mf1


def clip_gradient(model, clip_value=1e-1):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def init_logfile_nn(method_name, opt):
    logfile = CSVLog(opt.logfile_gru, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} ' \
                                                          f'and run {opt.seed} already calculated'
    return logfile
