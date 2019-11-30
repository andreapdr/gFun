import numpy as np
import time
from data.embeddings import WordEmbeddings, embedding_matrix, WCE_matrix
from scipy.sparse import issparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers.StandardizeTransformer import StandardizeTransformer


def _sort_if_sparse(X):
    if issparse(X) and not X.has_sorted_indices:
        X.sort_indices()


def _joblib_transform_multiling(transformer, lX, n_jobs=-1):
    if n_jobs == 1:
        return {lang:transformer(lX[lang]) for lang in lX.keys()}
    else:
        langs = list(lX.keys())
        transformations = Parallel(n_jobs=n_jobs)(delayed(transformer)(lX[lang]) for lang in langs)
        return {lang: transformations[i] for i, lang in enumerate(langs)}


class TrivialRejector:
    def fit(self, X, y):
        self.cats = y.shape[1]
        return self

    def decision_function(self, X): return np.zeros((X.shape[0],self.cats))

    def predict(self, X): return np.zeros((X.shape[0],self.cats))

    def predict_proba(self, X): return np.zeros((X.shape[0],self.cats))

    def best_params(self): return {}


class FunnellingPolylingualClassifier:
    """
    This classifier projects each document d into a language-independent feature space where each dimension fi is the
    decision score phi_l(d,ci) of an auxiliary classifier phi_l trained on category ci for documents in language l;
    then trains one single classifier for all documents in this space, irrespective of their originary language
    """
    def __init__(self, first_tier_learner, meta_learner, first_tier_parameters=None, meta_parameters=None, folded_projections=1,
                 calmode='cal', n_jobs=-1):
        """
        :param first_tier_learner: the learner used in the first-tier level
        :param meta_learner: the learner used in the second-tier level
        :param first_tier_parameters: parameters for the learner in the doc_projector
        :param meta_parameters: parameters for the learner in the z-space
        :param folded_projections: if 1 then the model trains the auxiliar classifiers with all training data and
        :param calmode: 'cal' to calibrate the base classifiers, 'nocal' to use the decision_function instead, or
        :param n_jobs: number of parallel threads
        'sigmoid' to use the sigmoid of the decision_function
        projects the data before training the final classifier; if greater than one, the training set is split in as
        many folds as indicated, and the projected space is composed by concatenating each fold prediction based on
        models trained on the remaining folds. This should increase the generality of the space to unseen data.
        """
        assert folded_projections>0, "positive number of folds expected"
        assert calmode in ['cal','nocal','sigmoid'], 'unknown calmode'
        assert calmode!='cal' or first_tier_learner.probability, 'calmode=cal requires the learner to have probability=True'

        self.fist_tier_learner = first_tier_learner
        self.meta_learner = meta_learner
        self.fist_tier_parameters=first_tier_parameters
        self.meta_parameters = meta_parameters
        self.doc_projector = NaivePolylingualClassifier(self.fist_tier_learner, self.fist_tier_parameters, n_jobs=n_jobs)
        self.doc_projector_bu = NaivePolylingualClassifier(self.fist_tier_learner, self.fist_tier_parameters, n_jobs=n_jobs)
        self.folded_projections = folded_projections
        self.n_jobs = n_jobs
        self.calmode = calmode

    def _projection(self, doc_projector, lX):
        """
        Decides the projection function to be applied; predict_proba if the base classifiers are calibrated or
        decision_function if otherwise
        :param doc_projector: the document projector (a NaivePolylingualClassifier)
        :param lX: {lang:matrix} to train
        :return: the projection, applied with predict_proba or decision_function
        """
        if self.calmode=='cal':
            return doc_projector.predict_proba(lX)
        else:
            l_decision_scores = doc_projector.decision_function(lX)
            if self.calmode=='sigmoid':
                def sigmoid(x): return 1 / (1 + np.exp(-x))
                for lang in l_decision_scores.keys():
                    l_decision_scores[lang] = sigmoid(l_decision_scores[lang])
            return l_decision_scores

    def _get_zspace(self, lXtr, lYtr, lXproj=None, lYproj=None):
        """
        Produces the vector space of posterior probabilities (in case the first-tier is calibrated) or of
        decision scores (if otherwise). This space is here named zspace.
        :param lXtr: {lang:matrix} to train
        :param lYtr: {lang:labels} to train
        :param lXproj: {lang:matrix} to project (if None, then projects the lXtr)
        :param lYproj: {lang:labels} to stack in the same order (if None, then lYtr will be stacked)
        :return: the projection of lXproj documents into the Z-space defined by the confidence scores of language-specific
        models trained on lXtr, and the lYproj labels stacked consistently
        """
        repair_empty_folds = True
        if lXproj is None and lYproj is None:
            lXproj, lYproj = lXtr, lYtr
            repair_empty_folds = False

        print('fitting the projectors... {}'.format(lXtr.keys()))
        self.doc_projector.fit(lXtr, lYtr)

        print('projecting the documents')
        langs = list(lXtr.keys())
        lZ = self._projection(self.doc_projector, lXproj)

        # if repair_empty_folds: #empty folds are replaced by the posterior probabilities generated by the non-folded version
        empty_categories = self.doc_projector.empty_categories
        lZ_bu = self._projection(self.doc_projector_bu, lXproj)

        for lang in langs:
            repair = empty_categories[lang]
            lZ[lang][:,repair] = lZ_bu[lang][:,repair]

        Z = np.vstack([lZ[lang] for lang in langs])  # Z is the language independent space
        zy = np.vstack([lYproj[lang] for lang in langs])
        return Z, zy

    def _get_zspace_folds(self, lX, ly):
        self.doc_projector_bu.fit(lX, ly)

        print('split of {} folds'.format(self.folded_projections))
        skf = KFold(n_splits=self.folded_projections, shuffle=True)

        Z, zy = [], []
        lfold = {lang: list(skf.split(lX[lang], ly[lang])) for lang in lX.keys()}
        for fold in range(self.folded_projections):
            print('fitting the projectors ({}/{})...'.format(fold + 1, self.folded_projections))
            lfoldXtr, lfoldYtr = {}, {}
            lfoldXte, lfoldYte = {}, {}
            for lang in lX.keys():
                train, test = lfold[lang][fold]
                lfoldXtr[lang] = lX[lang][train]
                lfoldYtr[lang] = ly[lang][train]
                lfoldXte[lang] = lX[lang][test]
                lfoldYte[lang] = ly[lang][test]
            Zfold, zYfold = self._get_zspace(lfoldXtr, lfoldYtr, lfoldXte, lfoldYte)
            Z.append(Zfold)
            zy.append(zYfold)
        # compose the Z-space as the union of all folded predictions
        Z = np.vstack(Z)
        zy = np.vstack(zy)
        # refit the document projector with all examples to have a more reliable projector for test data
        self.doc_projector = self.doc_projector_bu
        return Z, zy

    def fit(self, lX, ly, lZ=None, lzy=None):
        tinit = time.time()
        Z, zy = self._get_zspace(lX, ly) if self.folded_projections == 1 else self._get_zspace_folds(lX, ly)

        #experimental: adds the posterior probabilities (computed outside) to the meta-classifier
        if lZ is not None and lzy is not None:
            zlangs = list(lZ.keys())
            Z = np.vstack((Z, *[lZ[l] for l in zlangs]))
            zy = np.vstack((zy, *[lzy[l] for l in zlangs]))

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model = MonolingualClassifier(base_learner=self.meta_learner, parameters=self.meta_parameters, n_jobs=self.n_jobs)
        self.model.fit(Z, zy)
        self.time = time.time() - tinit

        return self

    def predict(self, lX, lZ=None):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :param lZ: a dictionary {language_label: Z matrix}; if specified, concats this representation
        :return: a dictionary of predictions
        """
        lZ_ = self._projection(self.doc_projector, lX)
        if lZ is not None:
            lZ_ = {**lZ_, **lZ}
        return _joblib_transform_multiling(self.model.predict, lZ_, n_jobs=self.n_jobs)

    def best_params(self):
        params = self.doc_projector.best_params()
        params['meta'] = self.model.best_params()
        return params


class NaivePolylingualClassifier:
    """
    Is a mere set of independet MonolingualClassifiers
    """
    def __init__(self, base_learner, parameters=None, n_jobs=-1):
        self.base_learner = base_learner
        self.parameters = parameters
        self.model = None
        self.n_jobs = n_jobs

    def fit(self, lX, ly):
        """
        trains the independent monolingual classifiers
        :param lX: a dictionary {language_label: X csr-matrix}
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        langs = list(lX.keys())
        for lang in langs:
            _sort_if_sparse(lX[lang])

        models = Parallel(n_jobs=self.n_jobs)\
            (delayed(MonolingualClassifier(self.base_learner, parameters=self.parameters).fit)((lX[lang]),ly[lang]) for lang in langs)
        #
        # models = [MonolingualClassifier(self.base_learner, parameters=self.parameters) for lang in langs]
        #
        # for model, lang in zip(models, langs):
        #     model.fit(lX[lang], ly[lang])

        self.model = {lang: models[i] for i, lang in enumerate(langs)}
        self.empty_categories = {lang:self.model[lang].empty_categories for lang in langs}
        self.time = time.time() - tinit
        return self

    def decision_function(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of classification scores for each class
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in decision function'
        langs=list(lX.keys())
        scores = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].decision_function)(lX[lang]) for lang in langs)
        return {lang:scores[i] for i,lang in enumerate(langs)}

    def predict_proba(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of probabilities that each document belongs to each class
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in decision function'
        langs=list(lX.keys())
        scores = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(delayed(self.model[lang].predict_proba)(lX[lang]) for lang in langs)
        return {lang:scores[i] for i,lang in enumerate(langs)}

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in predict'
        if self.n_jobs == 1:
            return {lang:self.model[lang].predict(lX[lang]) for lang in lX.keys()}
        else:
            langs = list(lX.keys())
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].predict)(lX[lang]) for lang in langs)
            return {lang: scores[i] for i, lang in enumerate(langs)}

    def best_params(self):
        return {l:model.best_params() for l,model in self.model.items()}


class MonolingualClassifier:

    def __init__(self, base_learner, parameters=None, n_jobs=-1):
        self.learner = base_learner
        self.parameters = parameters
        self.model = None
        self.n_jobs = n_jobs
        self.best_params_ = None

    def fit(self, X, y):
        if X.shape[0] == 0:
            print('Warning: X has 0 elements, a trivial rejector will be created')
            self.model = TrivialRejector().fit(X,y)
            self.empty_categories = np.arange(y.shape[1])
            return self

        tinit = time.time()
        _sort_if_sparse(X)
        self.empty_categories = np.argwhere(np.sum(y, axis=0)==0).flatten()

        # multi-class format
        if len(y.shape) == 2:
            if self.parameters is not None:
                self.parameters = [{'estimator__' + key: params[key] for key in params.keys()}
                                   for params in self.parameters]
            self.model = OneVsRestClassifier(self.learner, n_jobs=self.n_jobs)
        else:
            self.model = self.learner
            raise NotImplementedError('not working as a base-classifier for funneling if there are gaps in the labels across languages')

        # parameter optimization?
        if self.parameters:
            print('debug: optimizing parameters:', self.parameters)
            self.model = GridSearchCV(self.model, param_grid=self.parameters, refit=True, cv=5, n_jobs=self.n_jobs,
                                      error_score=0, verbose=10)

        print('fitting:', self.model)
        self.model.fit(X, y)
        if isinstance(self.model, GridSearchCV):
            self.best_params_ = self.model.best_params_
            print('best parameters: ', self.best_params_)
        self.time=time.time()-tinit
        return self

    def decision_function(self, X):
        assert self.model is not None, 'predict called before fit'
        _sort_if_sparse(X)
        return self.model.decision_function(X)

    def predict_proba(self, X):
        assert self.model is not None, 'predict called before fit'
        assert hasattr(self.model, 'predict_proba'), 'the probability predictions are not enabled in this model'
        _sort_if_sparse(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        assert self.model is not None, 'predict called before fit'
        _sort_if_sparse(X)
        return self.model.predict(X)

    def best_params(self):
        return self.best_params_


class PolylingualEmbeddingsClassifier:
    """
    This classifier creates document embeddings by a tfidf weighted average of polylingual embeddings from the article
    @article{conneau2017word,
      title={Word translation without parallel data},
      author={Conneau, Alexis and Lample, Guillaume and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
      journal={arXiv preprint arXiv:1710.04087},
      year={2017}
    }
    url: https://github.com/facebookresearch/MUSE
    """
    def __init__(self, wordembeddings_path, learner, c_parameters=None, n_jobs=-1):
        """
        :param wordembeddings_path: the path to the directory containing the polylingual embeddings
        :param learner: the learner
        :param c_parameters: parameters for learner
        :param n_jobs: the number of concurrent threads
        """
        self.wordembeddings_path = wordembeddings_path
        self.learner = learner
        self.c_parameters=c_parameters
        self.n_jobs = n_jobs
        self.lang_tfidf = {}
        self.model = None

    def fit_vectorizers(self, lX):
        for lang in lX.keys():
            if lang not in self.lang_tfidf:
                tfidf = TfidfVectorizer(sublinear_tf=True, use_idf=True)  # text is already processed
                docs = lX[lang]
                tfidf.fit(docs)
                self.lang_tfidf[lang] = tfidf

    def embed(self, docs, lang):
        assert lang in self.lang_tfidf, 'unknown language'
        tfidf_vectorizer = self.lang_tfidf[lang]
        V = tfidf_vectorizer.vocabulary_
        Xweights = tfidf_vectorizer.transform(docs)

        print('loading word embeddings for ' + lang)
        we = WordEmbeddings.load(self.wordembeddings_path, lang)

        nD = len(docs)
        doc_vecs = np.zeros((nD, we.dim()))

        for i, doc in enumerate(docs):
            print('\r\tcomplete {:.3f}%'.format(100 * (i + 1) / nD), end='')
            # averaging with tfidf (summing each word only once, since the frequency is already controlled)
            for w in set(doc.split()):
                if w in we and w in V:
                    doc_vecs[i] += (we[w] * Xweights[i, V[w]])
            # works much worse with idf; works much worse with document l2-normalization
        print()

        return doc_vecs

    def fit(self, lX, ly):
        """
        :param lX: a dictionary {language_label: [list of preprocessed documents]}
        :param ly: a dictionary {language_label: ndarray of shape (ndocs, ncats) binary labels}
        :return: self
        """
        tinit = time.time()
        langs = list(lX.keys())
        WEtr, Ytr = [], []
        self.fit_vectorizers(lX) # if already fit, does nothing
        _lX = dict()
        for lang in langs:
            _lX[lang] = self.lang_tfidf[lang].transform(lX[lang])
            WEtr.append(self.embed(lX[lang], lang))
            Ytr.append(ly[lang])

        # TODO @Andrea --> here embeddings should be stacked horizontally!
        WEtr = np.vstack(WEtr)
        Ytr = np.vstack(Ytr)
        self.embed_time = time.time() - tinit

        print('fitting the WE-space of shape={}'.format(WEtr.shape))
        self.model = MonolingualClassifier(base_learner=self.learner, parameters=self.c_parameters, n_jobs=self.n_jobs)
        self.model.fit(_lX['da'], ly['da'])
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: [list of preprocessed documents]}
        """
        assert self.model is not None, 'predict called before fit'
        langs = list(lX.keys())
        lWEte = {lang:self.embed(lX[lang], lang) for lang in langs} # parallelizing this may consume too much memory
        return _joblib_transform_multiling(self.model.predict, lWEte, n_jobs=self.n_jobs)

    def predict_proba(self, lX):
        """
        :param lX: a dictionary {language_label: [list of preprocessed documents]}
        """
        assert self.model is not None, 'predict called before fit'
        langs = list(lX.keys())
        # lWEte = {lang:self.embed(lX[lang], lang) for lang in langs} # parallelizing this may consume too much memory
        return _joblib_transform_multiling(self.model.predict_proba, self.lang_tfidf['da'], n_jobs=self.n_jobs)

    def best_params(self):
        return self.model.best_params()


class AndreaCLF(FunnellingPolylingualClassifier):
    def __init__(self,
                 we_path,
                 config,
                 first_tier_learner,
                 meta_learner,
                 first_tier_parameters=None,
                 meta_parameters=None,
                 folded_projections=1,
                 calmode='cal',
                 n_jobs=-1):

        super().__init__(first_tier_learner,
                         meta_learner,
                         first_tier_parameters,
                         meta_parameters,
                         folded_projections,
                         calmode,
                         n_jobs)

        self.we_path = we_path
        self.config = config
        self.lang_word2idx = dict()
        self.languages = []
        self.lang_tfidf = {}
        self.word_embeddings = {}
        self.supervised_embeddings = {}
        self.model = None
        self.time = None

    def vectorize(self, lX, prediction=False):
        langs = list(lX.keys())
        print(f'# tfidf-vectorizing docs')
        if prediction:
            for lang in langs:
                assert lang in self.lang_tfidf.keys(), 'no tf-idf for given language'
                tfidf_vectorizer = self.lang_tfidf[lang]
                lX[lang] = tfidf_vectorizer.transform(lX[lang])
            return self

        for lang in langs:
            tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
            self.languages.append(lang)
            tfidf_vectorizer.fit(lX[lang])
            lX[lang] = tfidf_vectorizer.transform(lX[lang])
            self.lang_word2idx[lang] = tfidf_vectorizer.vocabulary_
            self.lang_tfidf[lang] = tfidf_vectorizer
        return self

    # @override std class method
    def _get_zspace(self, lXtr, lYtr):
        print('\nfitting the projectors... {}'.format(list(lXtr.keys())))
        self.doc_projector.fit(lXtr, lYtr)

        print('\nprojecting the documents')
        lZ = self._projection(self.doc_projector, lXtr)

        return lZ, lYtr

    def embed(self, lX, ly, unsupervised=False, supervised=False, prediction=False):
        """
        build embedding matrix for given language and returns its weighted sum wrt tf-idf score
        """
        _r = dict()
        languages = list(lX.keys())

        if prediction:
            for lang in languages:
                if unsupervised:    # If unsupervised embeddings ...
                    M = self.word_embeddings[lang]
                    if supervised:  # and also unsupervised --> get both (M) and (S) weighted sum matrices and hstack them
                        S = self.supervised_embeddings[lang]
                        _r[lang] = np.hstack((lX[lang].dot(M), lX[lang].dot(S)))
                        continue
                    _r[lang] = lX[lang].dot(M)  # if not supervised --> just get weighted sum of unsupervised (M) embeddings
                else:   # If not unsupervised --> get (S) matrix and its weighted sum 
                    S = self.supervised_embeddings[lang]
                    _r[lang] = lX[lang].dot(S)
            return _r

        if unsupervised:
            for lang in languages:
                _, M = embedding_matrix(self.config['we_type'], self.we_path, self.lang_word2idx[lang], lang)
                self.word_embeddings[lang] = M
                _r[lang] = lX[lang].dot(M)

        if supervised:
            for lang in languages:
                S = WCE_matrix(lX, ly, lang)
                self.supervised_embeddings[lang] = S
                if unsupervised:
                    _r[lang] = np.hstack((_r[lang], lX[lang].dot(S)))
                else:
                    _r[lang] = lX[lang].dot(S)
        return _r

    # @override std class method
    def fit(self, lX, ly):
        tinit = time.time()
        print('Vectorizing documents...')
        self.vectorize(lX)

        for lang in self.languages:
            print(f'{lang}->{lX[lang].shape}')

        Z, zy = self._get_zspace(lX, ly)

        if self.config['supervised'] or self.config['unsupervised']:
            # Z vectors is concatenated with doc's embedding weighted sum
            Z_embedded = dict()
            l_weighted_em = self.embed(lX, ly,
                                       unsupervised=self.config['unsupervised'],
                                       supervised=self.config['supervised'])

            # stacking Z space horizontally with unsupervised (M) and/or supervised (F) embeddings
            for lang in list(lX.keys()):
                Z_embedded[lang] = np.hstack((Z[lang], l_weighted_em[lang]))
            Z = Z_embedded


        # stacking Z space vertically
        _vertical_Z = np.vstack([Z[lang] for lang in self.languages])
        _vertical_Zy = np.vstack([zy[lang] for lang in self.languages])

        self.standardizer = StandardizeTransformer()
        _vertical_Z = self.standardizer.fit_predict(_vertical_Z)

        print('fitting the Z-space of shape={}'.format(_vertical_Z.shape))
        self.model = MonolingualClassifier(base_learner=self.meta_learner, parameters=self.meta_parameters,
                                           n_jobs=self.n_jobs)
        self.model.fit(_vertical_Z, _vertical_Zy)
        self.time = time.time() - tinit
        print(f'\nTotal training time elapsed: {round((self.time/60), 2)} min')

    def predict(self, lX, ly):
        print('Vectorizing documents')
        self.vectorize(lX, prediction=True)
        lZ = self._projection(self.doc_projector, lX)

        if self.config['supervised'] or self.config['unsupervised']:
            l_weighted_em = self.embed(lX, ly,
                                       unsupervised=self.config['unsupervised'],
                                       supervised=self.config['supervised'],
                                       prediction=True)
            Z_embedded = dict()
            for lang in lX.keys():
                Z_embedded[lang] = np.hstack((lZ[lang], l_weighted_em[lang]))
            lZ = Z_embedded

        for lang in lZ.keys():
            print(lZ[lang].shape)
            lZ[lang] = self.standardizer.predict(lZ[lang])

        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)
