import os
from torchtext.vocab import Vectors
import torch
from abc import ABC, abstractmethod
from util.SIF_embed import *


class PretrainedEmbeddings(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def vocabulary(self): pass

    @abstractmethod
    def dim(self): pass

    @classmethod
    def reindex(cls, words, word2index):
        if isinstance(words, dict):
            words = list(zip(*sorted(words.items(), key=lambda x: x[1])))[0]

        source_idx, target_idx = [], []
        for i, word in enumerate(words):
            if word not in word2index: continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        return source_idx, target_idx


class FastTextWikiNews(Vectors):

    url_base = 'Cant auto-download MUSE embeddings'
    path = '../embeddings/wiki.multi.{}.vec'
    _name = '/wiki.multi.{}.vec'

    def __init__(self, cache, language="en", **kwargs):
        url = self.url_base.format(language)
        name = cache + self._name.format(language)
        super(FastTextWikiNews, self).__init__(name, cache=cache, url=url, **kwargs)


class FastTextMUSE(PretrainedEmbeddings):
    def __init__(self, path, lang, limit=None):
        super().__init__()
        print(f'Loading fastText pretrained vectors for language {lang} from {path}')
        assert os.path.exists(path), print(f'pre-trained vectors not found in {path}')
        self.embed = FastTextWikiNews(path, lang, max_vectors=limit)

    def vocabulary(self):
        return set(self.embed.stoi.keys())

    def dim(self):
        return self.embed.dim

    def extract(self, words):
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.embed.stoi)
        extraction = torch.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        return extraction



