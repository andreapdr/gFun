import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from data.text_preprocessor import NLTKStemTokenizer
from data.tsr_function__ import get_tsr_matrix, get_supervised_matrix, pointwise_mutual_information, information_gain, \
    gain_ratio, gss
from embeddings.embeddings import FastTextMUSE
from embeddings.supervised import supervised_embeddings_tfidf, zscores
from learning.learners import NaivePolylingualClassifier, MonolingualClassifier, _joblib_transform_multiling
import time
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from scipy.sparse import issparse, vstack, hstack
from transformers.StandardizeTransformer import StandardizeTransformer
from util.SIF_embed import remove_pc
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from scipy.sparse import csr_matrix

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------


class TfidfVectorizerMultilingual:

    def __init__(self, **kwargs):
        self.kwargs=kwargs

    def fit(self, lX, ly=None):
        self.langs = sorted(lX.keys())
        self.vectorizer={l:TfidfVectorizer(**self.kwargs).fit(lX[l]) for l in self.langs}
        # tokenizer=NLTKStemTokenizer(l, verbose=True),
        return self

    def transform(self, lX):
        return {l:self.vectorizer[l].transform(lX[l]) for l in self.langs}

    def fit_transform(self, lX, ly=None):
        return self.fit(lX,ly).transform(lX)

    def vocabulary(self, l=None):
        if l is None:
            return {l:self.vectorizer[l].vocabulary_ for l in self.langs}
        else:
            return self.vectorizer[l].vocabulary_

    def get_analyzer(self, l=None):
        if l is None:
            return {l:self.vectorizer[l].build_analyzer() for l in self.langs}
        else:
            return self.vectorizer[l].build_analyzer()


class FeatureWeight:

    def __init__(self, weight='tfidf', agg='mean'):
        assert weight in ['tfidf', 'pmi', 'ig'] or callable(weight), 'weight should either be "tfidf" or a callable function'
        assert agg in ['mean', 'max'], 'aggregation function should either be "mean" or "max"'
        self.weight = weight
        self.agg = agg
        self.fitted = False
        if weight=='pmi':
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
# Document Embeddings
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

    def fit(self, lX, lY, lV=None):
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
        print(f'generating posterior probabilities for {sum([X.shape[0] for X in lX.values()])} the documents')
        return self.doc_projector.predict_proba(lX)


class MuseEmbedder:

    def __init__(self, path, lV=None, l2=True, n_jobs=-1, featureweight=FeatureWeight()):
        self.path=path
        self.lV = lV
        self.l2 = l2
        self.n_jobs = n_jobs
        self.featureweight = featureweight

    def fit(self, lX, ly, lV=None):
        assert lV is not None or self.lV is not None, 'lV not specified'
        self.langs = sorted(lX.keys())
        self.MUSE = load_muse_embeddings(self.path, self.langs, self.n_jobs)
        lWordList = {l:self._get_wordlist_from_word2index(lV[l]) for l in self.langs}
        self.MUSE = {l:Muse.extract(lWordList[l]).numpy() for l,Muse in self.MUSE}
        self.featureweight.fit(lX, ly)
        return self

    def transform(self, lX):
        MUSE = self.MUSE
        lX = self.featureweight.transform(lX)
        XdotMUSE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], MUSE[lang]) for lang in self.langs
        )
        lMuse = {l: XdotMUSE[i] for i, l in enumerate(self.langs)}
        lMuse = _normalize(lMuse, self.l2)
        return lMuse

    def fit_transform(self, lX, ly, lV):
        return self.fit(lX, ly, lV).transform(lX)

    def _get_wordlist_from_word2index(self, word2index):
        return list(zip(*sorted(word2index.items(), key=lambda x: x[1])))[0]


class WordClassEmbedder:

    def __init__(self, l2=True, n_jobs=-1, max_label_space=300, featureweight=FeatureWeight()):
        self.n_jobs = n_jobs
        self.l2 = l2
        self.max_label_space=max_label_space
        self.featureweight = featureweight

    def fit(self, lX, ly, lV=None):
        self.langs = sorted(lX.keys())
        WCE = Parallel(n_jobs=self.n_jobs)(
            delayed(word_class_embedding_matrix)(lX[lang], ly[lang], self.max_label_space) for lang in self.langs
        )
        self.lWCE = {l:WCE[i] for i,l in enumerate(self.langs)}
        self.featureweight.fit(lX, ly)
        return self

    def transform(self, lX):
        lWCE = self.lWCE
        lX = self.featureweight.transform(lX)
        XdotWCE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], lWCE[lang])for lang in self.langs
        )
        lwce = {l: XdotWCE[i] for i, l in enumerate(self.langs)}
        lwce = _normalize(lwce, self.l2)
        return lwce

    def fit_transform(self, lX, ly, lV=None):
        return self.fit(lX, ly).transform(lX)


class DocEmbedderList:

    def __init__(self, *embedder_list, aggregation='concat'):
        assert aggregation in {'concat', 'mean'}, 'unknown aggregation mode, valid are "concat" and "mean"'
        if len(embedder_list)==0: embedder_list=[]
        self.embedders = embedder_list
        self.aggregation = aggregation

    def fit(self, lX, ly, lV=None):
        for transformer in self.embedders:
            transformer.fit(lX,ly,lV)
        return self

    def transform(self, lX):
        if self.aggregation == 'concat':
            return self.transform_concat(lX)
        elif self.aggregation == 'mean':
            return self.transform_mean(lX)

    def transform_concat(self, lX):
        if len(self.embedders)==1:
            return self.embedders[0].transform(lX)

        some_sparse = False
        langs = sorted(lX.keys())

        lZparts = {l: [] for l in langs}
        for transformer in self.embedders:
            lZ = transformer.transform(lX)
            for l in langs:
                Z = lZ[l]
                some_sparse = some_sparse or issparse(Z)
                lZparts[l].append(Z)

        hstacker = hstack if some_sparse else np.hstack
        return {l:hstacker(lZparts[l]) for l in langs}

    def transform_mean(self, lX):
        if len(self.embedders)==1:
            return self.embedders[0].transform(lX)

        langs = sorted(lX.keys())

        lZparts = {l: None for l in langs}
        for transformer in self.embedders:
            lZ = transformer.transform(lX)
            for l in langs:
                Z = lZ[l]
                if lZparts[l] is None:
                    lZparts[l] = Z
                else:
                    lZparts[l] += Z

        n_transformers = len(self.embedders)

        return {l:lZparts[l] / n_transformers for l in langs}

    def fit_transform(self, lX, ly, lV=None):
        return self.fit(lX, ly, lV).transform(lX)

    def best_params(self):
        return {'todo'}

    def append(self, embedder):
        self.embedders.append(embedder)


class FeatureSet2Posteriors:
    def __init__(self, transformer, l2=True, n_jobs=-1):
        self.transformer = transformer
        self.l2=l2
        self.n_jobs = n_jobs
        self.prob_classifier = MetaClassifier(SVC(kernel='rbf', probability=True, cache_size=1000, random_state=1), n_jobs=n_jobs)

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
# Meta-Classifier
# ------------------------------------------------------------------
class MetaClassifier:

    def __init__(self, meta_learner, meta_parameters=None, n_jobs=-1, standardize_range=None):
        self.n_jobs=n_jobs
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
# Ensembling
# ------------------------------------------------------------------
class Funnelling:
    def __init__(self,
                 vectorizer:TfidfVectorizerMultilingual,
                 first_tier:DocEmbedderList,
                 meta:MetaClassifier):
        self.vectorizer = vectorizer
        self.first_tier = first_tier
        self.meta = meta
        self.n_jobs = meta.n_jobs

    def fit(self, lX, ly):
        lX = self.vectorizer.fit_transform(lX, ly)
        lV = self.vectorizer.vocabulary()
        lZ = self.first_tier.fit_transform(lX, ly, lV)
        self.meta.fit(lZ, ly)

    def predict(self, lX, ly=None):
        lX = self.vectorizer.transform(lX)
        lZ = self.first_tier.transform(lX)
        ly_ = self.meta.predict(lZ)
        return ly_

    def best_params(self):
        return {'1st-tier':self.first_tier.best_params(),
                'meta':self.meta.best_params()}


class Voting:
    def __init__(self, *prob_classifiers):
        assert all([hasattr(p, 'predict_proba') for p in prob_classifiers]), 'not all classifiers are probabilistic'
        self.prob_classifiers = prob_classifiers

    def fit(self, lX, ly, lV=None):
        for classifier in self.prob_classifiers:
            classifier.fit(lX, ly, lV)

    def predict(self, lX, ly=None):

        lP = {l:[] for l in lX.keys()}
        for classifier in self.prob_classifiers:
            lPi = classifier.predict_proba(lX)
            for l in lX.keys():
                lP[l].append(lPi[l])

        lP = {l:np.stack(Plist).mean(axis=0) for l,Plist in lP.items()}
        ly = {l:P>0.5 for l,P in lP.items()}

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
    print('computing supervised embeddings...')
    WCE = supervised_embeddings_tfidf(X, Y)
    WCE = zscores(WCE, axis=0)

    nC = Y.shape[1]
    if nC > max_label_space:
        print(f'supervised matrix has more dimensions ({nC}) than the allowed limit {max_label_space}. '
              f'Applying PCA(n_components={max_label_space})')
        pca = PCA(n_components=max_label_space)
        WCE = pca.fit(WCE).transform(WCE)

    return WCE


def XdotM(X,M):
    # return X.dot(M)
    # print(f'X={X.shape}, M={M.shape}')
    E = X.dot(M)
    E = remove_pc(E, npc=1)
    return E


def _normalize(lX, l2=True):
   return {l: normalize(X) for l, X in lX.items()} if l2 else lX


