import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import cosine
import operator
import functools
import math, sys
# from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed


class DistributionalCorrespondenceIndexing:

    prob_dcf = ['linear', 'pmi']
    vect_dcf = ['cosine']
    valid_dcf = prob_dcf + vect_dcf
    valid_post = ['normal', 'l2', None]

    def __init__(self, dcf='cosine', post='normal', n_jobs=-1):
        """
        :param dcf: a distributional correspondence function name (e.g., 'cosine') or a callable f(u,v) which measures
                the distribucional correspondence between vectors u and v
        :param post: post-processing function to apply to document embeddings. Default is to standardize it into a
                normal distribution; other functions allowed are 'l2' or None
        """
        if post not in self.valid_post:
            raise ValueError("unknown post processing function; valid ones are [%s]" % ', '.join(self.valid_post))

        if isinstance(dcf, str):
            if dcf not in self.valid_dcf:
                raise ValueError("unknown dcf; use any in [%s]" % ', '.join(self.valid_dcf))
            self.dcf = getattr(DistributionalCorrespondenceIndexing, dcf)
        elif hasattr(dcf, '__call__'):
            self.dcf = dcf
        else:
            raise ValueError('param dcf should either be a valid dcf name in [%s] or a callable comparing two vectors')
        #self.dcf = lambda u,v:dcf(u,v)
        self.post = post
        self.domains = None
        self.dFP = None
        self.n_jobs = n_jobs

    def fit(self, dU, dP):
        """
        :param dU: a dictionary of {domain:dsm_matrix}, where dsm is a document-by-term matrix representing the
                distributional semantic model for a specific domain
        :param dP: a dictionary {domain:pivot_matrix} where domain is a string representing each domain,
                and pivot_matrix has shape (d,p) with d the dimensionality of the distributional space, and p the
                number of pivots
        :return: self
        """
        self.domains = list(dP.keys())
        assert len(np.unique([P.shape[1] for P in dP.values()]))==1, "inconsistent number of pivots across domains"
        assert set(dU.keys())==set(self.domains), "inconsistent domains in dU and dP"
        assert not [1 for d in self.domains if dU[d].shape[0]!=dP[d].shape[0]], \
            "inconsistent dimensions between distributional and pivot spaces"
        self.dimensions = list(dP.values())[0].shape[1]
        # embed the feature space from each domain using the pivots of that domain
        #self.dFP = {d:self.dcf_dist(dU[d].transpose(), dP[d].transpose()) for d in self.domains}
        transformations = Parallel(n_jobs=self.n_jobs)(delayed(self.dcf_dist)(dU[d].transpose(),dP[d].transpose()) for d in self.domains)
        self.dFP = {d: transformations[i] for i, d in enumerate(self.domains)}

    def _dom_transform(self, X, FP):
        _X = X.dot(FP)
        if self.post == 'l2':
            _X = normalize(_X, norm='l2', axis=1)
        elif self.post == 'normal':
            std = np.clip(np.std(_X, axis=0), 1e-5, None)
            _X = (_X - np.mean(_X, axis=0)) / std
        return _X

    # dX is a dictionary of {domain:dsm}, where dsm (distributional semantic model) is, e.g., a document-by-term csr_matrix
    def transform(self, dX):
        assert self.dFP is not None, 'transform method called before fit'
        assert set(dX.keys()).issubset(self.domains), 'domains in dX are not scope'
        domains = list(dX.keys())
        transformations = Parallel(n_jobs=self.n_jobs)(delayed(self._dom_transform)(dX[d], self.dFP[d]) for d in domains)
        return {d: transformations[i] for i, d in enumerate(domains)}

    def fit_transform(self, dU, dP, dX):
        return self.fit(dU, dP).transform(dX)

    def _prevalence(self, v):
        if issparse(v):
            return float(v.nnz) / functools.reduce(operator.mul, v.shape, 1) #this works for arrays of any rank
        elif isinstance(v, np.ndarray):
            return float(v[v>0].size) / v.size

    def linear(self, u, v, D):
        tp, fp, fn, tn = self._get_4cellcounters(u, v, D)
        den1=tp+fn
        den2=tn+fp
        tpr = (tp*1./den1) if den1!=0 else 0.
        tnr = (tn*1./den2) if den2!=0 else 0.
        return tpr + tnr - 1

    def pmi(self, u, v, D):
        tp, fp, fn, tn = self._get_4cellcounters(u, v, D)

        Pxy = tp * 1. / D
        Pxny = fp * 1. / D
        Pnxy = fn * 1. / D
        Px = Pxy + Pxny
        Py = Pxy + Pnxy

        if (Px == 0 or Py == 0 or Pxy == 0):
            return 0.0

        score =  math.log2(Pxy / (Px * Py))
        if np.isnan(score) or np.isinf(score):
            print('NAN')
            sys.exit()
        return score

    def cosine(self, u, v):
        pu = self._prevalence(u)
        pv = self._prevalence(v)
        return cosine(u, v) - np.sqrt(pu * pv)

    def _get_4cellcounters(self, u, v, D):
        """
        :param u: a set of indexes with a non-zero value
        :param v: a set of indexes with a non-zero value
        :param D: the number of events (i.e., all posible indexes)
        :return: the 4-cell contingency values tp, fp, fn, tn)
        """
        common=u.intersection(v)
        tp = len(common)
        fp = len(u) - len(common)
        fn = len(v) - len(common)
        tn = D - (tp + fp + fn)
        return tp, fp, fn, tn

    def dcf_dist(self, U, V):
        nU,D = U.shape
        nV = V.shape[0]
        if issparse(U): U = U.toarray()
        if issparse(V): V = V.toarray()

        dists = np.zeros((nU, nV))
        if self.dcf.__name__ in self.prob_dcf:
            def hits_index(v):
                return set(np.argwhere(v>0).reshape(-1).tolist())
            Vhits = {i:hits_index(V[i]) for i in range(nV)}
            for i in range(nU):
                Ui_hits = hits_index(U[i])
                for j in range(nV):
                    dists[i, j] = self.dcf(self, Ui_hits, Vhits[j], D)
        else:
            for i in range(nU):
                for j in range(nV):
                    dists[i, j] = self.dcf(self, U[i], V[j])
        return dists

