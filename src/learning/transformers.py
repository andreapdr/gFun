import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from data.text_preprocessor import NLTKStemTokenizer
from embeddings.supervised import supervised_embeddings_tfidf, zscores
from learning.learners import NaivePolylingualClassifier, MonolingualClassifier, _joblib_transform_multiling
import time
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from scipy.sparse import issparse, vstack, hstack
from transformers.StandardizeTransformer import StandardizeTransformer
from util.SIF_embed import remove_pc

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


# ------------------------------------------------------------------
# Document Embeddings
# ------------------------------------------------------------------
class PosteriorProbabilitiesEmbedder:

    def __init__(self, first_tier_learner, first_tier_parameters,
                 n_jobs=-1):
        self.fist_tier_learner = first_tier_learner
        self.fist_tier_parameters = first_tier_parameters
        self.n_jobs = n_jobs
        self.doc_projector = NaivePolylingualClassifier(self.fist_tier_learner,
                                                        self.fist_tier_parameters,
                                                        n_jobs=n_jobs)

    def fit(self, lX, lY):
        print('fitting the projectors... {}'.format(lX.keys()))
        self.doc_projector.fit(lX, lY)
        return self

    def transform(self, lX):
        print('projecting the documents')
        lZ = self.doc_projector.predict_proba(lX)
        return lZ

    def fit_transform(self, lX, ly=None):
        return self.fit(lX, ly).transform(lX)

    def best_params(self):
        return self.doc_projector.best_params()


class WordClassEmbedder:

    def __init__(self, n_jobs=-1, max_label_space=300):
        self.n_jobs = n_jobs
        self.max_label_space=max_label_space

    def fit(self, lX, ly):
        self.langs = sorted(lX.keys())
        WCE = Parallel(n_jobs=self.n_jobs)(
            delayed(word_class_embedding_matrix)(lX[lang], ly[lang], self.max_label_space) for lang in self.langs
        )
        self.lWCE = {l:WCE[i] for i,l in enumerate(self.langs)}
        return self

    def transform(self, lX):
        lWCE = self.lWCE
        XdotWCE = Parallel(n_jobs=self.n_jobs)(
            delayed(XdotM)(lX[lang], lWCE[lang]) for lang in self.langs
        )
        return {l: XdotWCE[i] for i, l in enumerate(self.langs)}

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


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
    E = X.dot(M)
    E = remove_pc(E, npc=1)
    return E


class DocEmbedderList:
    def __init__(self, *embedder_list):
        self.embedders = embedder_list

    def fit(self, lX, ly):
        for transformer in self.embedders:
            transformer.fit(lX,ly)
        return self

    def transform(self, lX):
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


    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)

    def best_params(self):
        return {'todo'}

# ------------------------------------------------------------------
# Meta-Classifier
# ------------------------------------------------------------------
class MetaClassifier:

    def __init__(self, meta_learner, meta_parameters, n_jobs=-1):
        self.n_jobs=n_jobs
        self.model = MonolingualClassifier(base_learner=meta_learner, parameters=meta_parameters, n_jobs=n_jobs)

    def fit(self, lZ, ly):
        tinit = time.time()
        Z, y = self.stack(lZ, ly)
        self.standardizer = StandardizeTransformer()
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
        lZ = self.first_tier.fit_transform(lX, ly)
        self.meta.fit(lZ, ly)

    def predict(self, lX, ly=None):
        lX = self.vectorizer.transform(lX)
        lZ = self.first_tier.transform(lX)
        ly_ = self.meta.predict(lZ)
        return ly_

    def best_params(self):
        return {'1st-tier':self.first_tier.best_params(),
                'meta':self.meta.best_params()}

