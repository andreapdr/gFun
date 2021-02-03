from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from src.data.languages import NLTK_LANGMAP


def preprocess_documents(documents, lang):
    tokens = NLTKStemTokenizer(lang, verbose=True)
    sw = stopwords.words(NLTK_LANGMAP[lang])
    return [' '.join([w for w in tokens(doc) if w not in sw]) for doc in documents]


class NLTKStemTokenizer(object):

    def __init__(self, lang, verbose=False):
        if lang not in NLTK_LANGMAP:
            raise ValueError('Language %s is not supported in NLTK' % lang)
        self.verbose=verbose
        self.called = 0
        self.wnl = SnowballStemmer(NLTK_LANGMAP[lang])
        self.cache = {}

    def __call__(self, doc):
        self.called += 1
        if self.verbose:
            print("\r\t\t[documents processed %d]" % (self.called), end="")
        tokens = word_tokenize(doc)
        stems = []
        for t in tokens:
            if t not in self.cache:
                self.cache[t] = self.wnl.stem(t)
            stems.append(self.cache[t])
        return stems