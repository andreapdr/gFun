import time

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from src.util.standardizer import StandardizeTransformer


def get_learner(calibrate=False, kernel='linear', C=1):
    """
    instantiate scikit Support Vector Classifier
    :param calibrate: boolean, whether to return posterior probabilities or not
    :param kernel: string,kernel to be applied to the SVC
    :param C: int or dict {'C': list of integer}, Regularization parameter
    :return: Support Vector Classifier
    """
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=C, random_state=1, gamma='auto', verbose=False)


def _sort_if_sparse(X):
    if issparse(X) and not X.has_sorted_indices:
        X.sort_indices()


def _joblib_transform_multiling(transformer, lX, n_jobs=-1):
    if n_jobs == 1:
        return {lang: transformer(lX[lang]) for lang in lX.keys()}
    else:
        langs = list(lX.keys())
        transformations = Parallel(n_jobs=n_jobs)(delayed(transformer)(lX[lang]) for lang in langs)
        return {lang: transformations[i] for i, lang in enumerate(langs)}


class TrivialRejector:
    def fit(self, X, y):
        self.cats = y.shape[1]
        return self

    def decision_function(self, X): return np.zeros((X.shape[0], self.cats))

    def predict(self, X): return np.zeros((X.shape[0], self.cats))

    def predict_proba(self, X): return np.zeros((X.shape[0], self.cats))

    def best_params(self): return {}


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
            (delayed(MonolingualClassifier(self.base_learner, parameters=self.parameters).fit)((lX[lang]), ly[lang]) for
             lang in langs)

        self.model = {lang: models[i] for i, lang in enumerate(langs)}
        self.empty_categories = {lang: self.model[lang].empty_categories for lang in langs}
        self.time = time.time() - tinit
        return self

    def decision_function(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of classification scores for each class
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in decision function'
        langs = list(lX.keys())
        scores = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].decision_function)(lX[lang]) for lang in langs)
        return {lang: scores[i] for i, lang in enumerate(langs)}

    def predict_proba(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of probabilities that each document belongs to each class
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in decision function'
        langs = list(lX.keys())
        scores = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(self.model[lang].predict_proba)(lX[lang]) for lang in langs)
        return {lang: scores[i] for i, lang in enumerate(langs)}

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in predict'
        if self.n_jobs == 1:
            return {lang: self.model[lang].transform(lX[lang]) for lang in lX.keys()}
        else:
            langs = list(lX.keys())
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].predict)(lX[lang]) for lang in langs)
            return {lang: scores[i] for i, lang in enumerate(langs)}

    def best_params(self):
        return {lang: model.best_params() for lang, model in self.model.items()}


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
            self.model = TrivialRejector().fit(X, y)
            self.empty_categories = np.arange(y.shape[1])
            return self

        tinit = time.time()
        _sort_if_sparse(X)
        self.empty_categories = np.argwhere(np.sum(y, axis=0) == 0).flatten()
        # multi-class format
        if len(y.shape) == 2:
            if self.parameters is not None:
                self.parameters = [{'estimator__' + key: params[key] for key in params.keys()}
                                   for params in self.parameters]
            self.model = OneVsRestClassifier(self.learner, n_jobs=self.n_jobs)
        else:
            self.model = self.learner
            raise NotImplementedError('not working as a base-classifier for funneling if there are gaps in '
                                      'the labels across languages')

        # parameter optimization?
        if self.parameters:
            print('debug: optimizing parameters:', self.parameters)
            self.model = GridSearchCV(self.model, param_grid=self.parameters, refit=True, cv=5, n_jobs=self.n_jobs,
                                      error_score=0, verbose=10)

        print(f'fitting: Mono-lingual Classifier on matrices of shape X={X.shape} Y={y.shape}')
        self.model.fit(X, y)
        if isinstance(self.model, GridSearchCV):
            self.best_params_ = self.model.best_params_
            print('best parameters: ', self.best_params_)
        self.time = time.time() - tinit
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
        Z = np.vstack([lZ[lang] for lang in langs])
        if ly is not None:
            y = np.vstack([ly[lang] for lang in langs])
            return Z, y
        else:
            return Z

    def predict(self, lZ):
        lZ = _joblib_transform_multiling(self.standardizer.transform, lZ, n_jobs=self.n_jobs)
        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)

    def predict_proba(self, lZ):
        lZ = _joblib_transform_multiling(self.standardizer.transform, lZ, n_jobs=self.n_jobs)
        return _joblib_transform_multiling(self.model.predict_proba, lZ, n_jobs=self.n_jobs)

