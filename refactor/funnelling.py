from models.learners import *
from view_generators import VanillaFunGen
from util.common import _normalize


class DocEmbedderList:
    def __init__(self, embedder_list, probabilistic=True):
        """
        Class that takes care of calling fit and transform function for every init embedder.
        :param embedder_list: list of embedders to be deployed
        :param probabilistic: whether to recast view generators output to vectors of posterior probabilities or not
        """
        assert len(embedder_list) != 0, 'Embedder list cannot be empty!'
        self.embedders = embedder_list
        self.probabilistic = probabilistic
        if probabilistic:
            _tmp = []
            for embedder in self.embedders:
                if isinstance(embedder, VanillaFunGen):
                    _tmp.append(embedder)
                else:
                    _tmp.append(FeatureSet2Posteriors(embedder))
        self.embedders = _tmp

    def fit(self, lX, ly):
        for embedder in self.embedders:
            embedder.fit(lX, ly)
        return self

    def transform(self, lX):
        langs = sorted(lX.keys())
        lZparts = {lang: None for lang in langs}

        for embedder in self.embedders:
            lZ = embedder.transform(lX)
            for lang in langs:
                Z = lZ[lang]
                if lZparts[lang] is None:
                    lZparts[lang] = Z
                else:
                    lZparts[lang] += Z
        n_embedders = len(self.embedders)
        return {lang: lZparts[lang]/n_embedders for lang in langs}

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class FeatureSet2Posteriors:
    def __init__(self, embedder, l2=True, n_jobs=-1):
        self.embedder = embedder
        self.l2 = l2
        self.n_jobs = n_jobs
        self.prob_classifier = MetaClassifier(
            SVC(kernel='rbf', gamma='auto', probability=True, cache_size=1000, random_state=1), n_jobs=n_jobs)

    def fit(self, lX, ly):
        lZ = self.embedder.fit_transform(lX, ly)
        self.prob_classifier.fit(lZ, ly)
        return self

    def transform(self, lX):
        lP = self.predict_proba(lX)
        lP = _normalize(lP, self.l2)
        return lP

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)

    def predict(self, lX):
        lZ = self.embedder.transform(lX)
        return self.prob_classifier.predict(lZ)

    def predict_proba(self, lX):
        lZ = self.embedder.transform(lX)
        return self.prob_classifier.predict_proba(lZ)


class Funnelling:
    def __init__(self, first_tier: DocEmbedderList, n_jobs=-1):
        self.first_tier = first_tier
        self.meta = MetaClassifier(
            SVC(kernel='rbf', gamma='auto', probability=True, cache_size=1000, random_state=1), n_jobs=n_jobs)
        self.n_jobs = n_jobs

    def fit(self, lX, ly):
        print('## Fitting first-tier learners!')
        lZ = self.first_tier.fit_transform(lX, ly)
        print('## Fitting meta-learner!')
        self.meta.fit(lZ, ly)

    def predict(self, lX):
        lZ = self.first_tier.transform(lX)
        ly = self.meta.predict(lZ)
        return ly
