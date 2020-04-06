import math
import numpy as np
from scipy.sparse import csr_matrix, issparse

class RandomIndexingBoC(object):

    def __init__(self, latent_dimensions, non_zeros=2):
        self.latent_dimensions = latent_dimensions
        self.k = non_zeros
        self.ri_dict = None

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        nF = X.shape[1]
        nL = self.latent_dimensions
        format = 'csr' if issparse(X) else 'np'
        self.ri_dict = _create_random_index_dictionary(shape=(nF, nL), k=self.k, normalized=True, format=format)
        return self

    def transform(self, X):
        assert X.shape[1] == self.ri_dict.shape[0], 'feature space is inconsistent with the RI dictionary'
        if self.ri_dict is None:
            raise ValueError("Error: transform method called before fit.")
        P = X.dot(self.ri_dict)
        if issparse(P):
            P.sort_indices()
        return P


def _create_random_index_dictionary(shape, k, normalized=False, format='csr', positive=False):
    assert format in ['csr', 'np'], 'Format should be in "[csr, np]"'
    nF, latent_dimensions = shape
    print("Creating the random index dictionary for |V|={} with {} dimensions".format(nF,latent_dimensions))
    val = 1.0 if not normalized else 1.0/math.sqrt(k)
    #ri_dict = csr_matrix((nF, latent_dimensions))  if format == 'csr' else np.zeros((nF, latent_dimensions))
    ri_dict = np.zeros((nF, latent_dimensions))

    #TODO: optimize
    for t in range(nF):
        dims = np.zeros(k, dtype=np.int32)
        dims[0] = t % latent_dimensions #the first dimension is choosen in a round-robin manner (prevents gaps)
        dims[1:] = np.random.choice(latent_dimensions, size=k-1, replace=False)
        values = (np.random.randint(0,2, size=k)*2.0-1.0) * val if not positive else np.array([+val]*k)
        ri_dict[t,dims]=values
        print("\rprogress [%.2f%% complete]" % (t * 100.0 / nF), end='')
    print('\nDone')

    if format=='csr':
        ri_dict = csr_matrix(ri_dict)
    return ri_dict

