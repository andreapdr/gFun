import numpy as np
import sklearn
# from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed

class ESA(object):
    """
    Implementation of Explicit Sematic Analysis (ESA) in its mono-lingual version, as a transformer
    """
    supported_similarity = ['dot', 'cosine']

    def __init__(self, similarity='dot', centered=False, post=None):
        """
        :param similarity: the similarity measure between documents to be used
        :param centered: set to True to subtract the expected similarity due to randomness (experimental)
        :param post: any valid sklearn normalization method to be applied to the resulting doc embeddings, or None (default)
        """
        assert similarity in self.supported_similarity, ("Similarity method %s is not supported" % similarity)
        self.similarity = similarity
        self.centered = centered
        self.post_processing = post
        self.W = None

    def fit(self, W):
        """
        :param W: doc-by-term already processed matrix of wikipedia documents
        :return: self
        """
        self.W = W
        return self

    def transform(self, X):
        """
        :param X: doc-by-term matrix that is to be transformed into the ESA space.
        :return: the matrix X transformed into the ESA space in numpy format
        """
        assert self.W is not None, 'transform method called before fit'

        W = self.W
        assert X.shape[1] == W.shape[1], ('the feature spaces for X=%s and W=%s do not agree' % (str(X.shape), str(W.shape)))

        if self.similarity in ['dot', 'cosine']:
            if self.similarity == 'cosine':
                X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True)
                W = sklearn.preprocessing.normalize(W, norm='l2', axis=1, copy=True)

            esa = (X.dot(W.T)).toarray()
            if self.centered:
                pX = (X > 0).sum(1) / float(X.shape[1])
                pW = (W > 0).sum(1) / float(W.shape[1])
                pXpW = np.sqrt(pX.dot(pW.transpose()))
                esa = esa - pXpW

            if self.post_processing:
                esa = sklearn.preprocessing.normalize(esa, norm=self.post_processing, axis=1, copy=True)

            return esa

    def fit_transform(self, W, X, Y=None):
        self.fit(W)
        return self.transform(X, Y)

    def dimensionality(self):
        return self.W.shape[0]



class CLESA(ESA):
    """
    Implementation of Cross-Lingual Explicit Sematic Analysis (ESA) as a transformer
    """
    
    def __init__(self, similarity='dot', centered=False, post=False, n_jobs=-1):
        super(CLESA, self).__init__(similarity, centered, post)
        self.lESA = None
        self.langs = None
        self.n_jobs = n_jobs

    def fit(self, lW):
        """
        :param lW: a dictionary of {language: doc-by-term wiki matrix}
        :return: self
        """
        assert len(np.unique([W.shape[0] for W in lW.values()])) == 1, "inconsistent dimensions across languages"

        self.dimensions = list(lW.values())[0].shape[0]
        self.langs = list(lW.keys())
        self.lESA = {lang:ESA(self.similarity, self.centered, self.post_processing).fit(lW[lang]) for lang in self.langs}
        return self

    def transform(self, lX):
        """
        :param lX: dictionary of {language : doc-by-term matrix} that is to be transformed into the CL-ESA space
        :return: a dictionary {language : doc-by-dim matrix} containing the matrix-transformed versions
        """
        assert self.lESA is not None, 'transform method called before fit'
        assert set(lX.keys()).issubset(set(self.langs)), 'languages in lX are not scope'
        langs = list(lX.keys())
        trans = Parallel(n_jobs=self.n_jobs)(delayed(self.lESA[lang].transform)(lX[lang]) for lang in langs)
        return {lang:trans[i] for i,lang in enumerate(langs)}

    def fit_transform(self, lW, lX):
        return self.fit(lW).transform(lX)

    def languages(self):
        return list(self.lESA.keys())




