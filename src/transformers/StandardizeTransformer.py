import numpy as np

class StandardizeTransformer:

    def __init__(self, axis=0):
        self.axis = axis
        self.yetfit=False

    def fit(self, X):
        print('fitting Standardizer')
        std=np.std(X, axis=self.axis, ddof=1)
        self.std = np.clip(std, 1e-5, None)
        self.mean = np.mean(X, axis=self.axis)
        self.yetfit=True
        print('done')
        return self

    def predict(self, X):
        if not self.yetfit: 'transform called before fit'
        return (X - self.mean) / self.std

    def fit_predict(self, X):
        return self.fit(X).predict(X)