from src.models.learners import *
from src.util.common import _normalize
from src.view_generators import VanillaFunGen


class DocEmbedderList:
    """
    Class that takes care of calling fit and transform function for every init embedder. Every ViewGenerator should be
    contained by this class in order to seamlessly train the overall architecture.
    """
    def __init__(self, embedder_list, probabilistic=True):
        """
        Init the DocEmbedderList.
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
        """
        Fit all the ViewGenerators contained by DocEmbedderList.
        :param lX:
        :param ly:
        :return: self
        """
        for embedder in self.embedders:
            embedder.fit(lX, ly)
        return self

    def transform(self, lX):
        """
        Project documents by means of every ViewGenerators. Projections are then averaged together and returned.
        :param lX:
        :return: common latent space (averaged).
        """
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
        return {lang: lZparts[lang]/n_embedders for lang in langs}  # Averaging feature spaces

    def fit_transform(self, lX, ly):
        return self.fit(lX, ly).transform(lX)


class FeatureSet2Posteriors:
    """
    Takes care of recasting features outputted by the embedders to vecotrs of posterior probabilities by means of
    a multiclass SVM.
    """
    def __init__(self, embedder, l2=True, n_jobs=-1):
        """
        Init the class.
        :param embedder: ViewGen, view generators which does not natively outputs posterior probabilities.
        :param l2: bool, whether to apply or not L2 normalization to the projection
        :param n_jobs: int, number of concurrent workers.
        """
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
    """
    Funnelling Architecture. It is composed by two tiers. The first-tier is a set of heterogeneous document embedders.
    The second-tier (i.e., the metaclassifier), operates the classification of the common latent space computed by
    the first-tier learners.
    """
    def __init__(self, first_tier: DocEmbedderList, meta_classifier: MetaClassifier, n_jobs=-1):
        self.first_tier = first_tier
        self.meta = meta_classifier
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
