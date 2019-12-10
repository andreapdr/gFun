from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def run_pca(dim, X):
    """
    :param dim: number of pca components to keep
    :param X: dictionary str(lang): matrix
    :return: dict lang: reduced matrix
    """
    r = dict()
    pca = PCA(n_components=dim)
    for lang in X.keys():
        r[lang] = pca.fit_transform(X[lang])
    return r


def get_optimal_dim(X, embed_type):
    """
    :param X: dict str(lang) : csr_matrix of embeddings unsupervised or supervised
    :param embed_type: (str) embedding matrix type: S or U (WCE supervised or U unsupervised MUSE/FASTTEXT)
    :return:
    """
    _idx = []

    plt.figure(figsize=(15, 10))
    if embed_type == 'U':
        plt.title(f'Unsupervised Embeddings {"TODO"} Explained Variance')
    else:
        plt.title(f'WCE Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')

    for lang in X.keys():
        pca = PCA(n_components=X[lang].shape[1])
        pca.fit(X[lang])
        _r = pca.explained_variance_ratio_
        _r = np.cumsum(_r)
        plt.plot(_r, label=lang)
        for i in range(len(_r) - 1, 1, -1):
            delta = _r[i] - _r[i - 1]
            if delta > 0:
                _idx.append(i)
                break
    best_n = max(_idx)
    plt.axvline(best_n, color='r', label='optimal N')
    plt.legend()
    plt.show()
    return best_n