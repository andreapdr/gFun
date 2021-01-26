import numpy as np


class StandardizeTransformer:
    def __init__(self, axis=0, range=None):
        """

        :param axis:
        :param range:
        """
        assert range is None or isinstance(range, slice), 'wrong format for range, should either be None or a slice'
        self.axis = axis
        self.yetfit = False
        self.range = range

    def fit(self, X):
        print('Applying z-score standardization...')
        std=np.std(X, axis=self.axis, ddof=1)
        self.std = np.clip(std, 1e-5, None)
        self.mean = np.mean(X, axis=self.axis)
        if self.range is not None:
            ones = np.ones_like(self.std)
            zeros = np.zeros_like(self.mean)
            ones[self.range] = self.std[self.range]
            zeros[self.range] = self.mean[self.range]
            self.std = ones
            self.mean = zeros
        self.yetfit=True
        return self

    def transform(self, X):
        if not self.yetfit: 'transform called before fit'
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)