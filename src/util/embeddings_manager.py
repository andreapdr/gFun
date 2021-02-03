from abc import ABC, abstractmethod

import numpy as np
import torch
from torchtext.vocab import Vectors

from src.util.SIF_embed import remove_pc


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
            if word not in word2index:
                continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        return source_idx, target_idx


class MuseLoader:
    def __init__(self, langs, cache):
        self.langs = langs
        self.lEmbed = {}
        self.lExtracted = {}
        for lang in self.langs:
            print(f'Loading vectors for {lang}...')
            self.lEmbed[lang] = Vectors(f'wiki.multi.{lang}.vec', cache)

    def dim(self):
        return self.lEmbed[list(self.lEmbed.keys())[0]].dim

    def vocabulary(self):
        return {lang: set(self.lEmbed[lang].stoi.keys()) for lang in self.langs}

    def extract(self, lVoc):
        """
        Reindex pretrained loaded embedding in order to match indexes assigned by scikit vectorizer. Such indexes
        are consistent with those used by Word Class Embeddings (since we deploy the same vectorizer)
        :param lVoc: dict {lang : {word : id}}
        :return: torch embedding matrix of extracted embeddings i.e., words in lVoc
        """
        for lang, words in lVoc.items():
            print(f'Extracting words for lang {lang}...')
            # words = list(zip(*sorted(lVoc[lang].items(), key=lambda x: x[1])))[0]
            source_id, target_id = PretrainedEmbeddings.reindex(words, self.lEmbed[lang].stoi)
            extraction = torch.zeros((len(words), self.dim()))
            extraction[source_id] = self.lEmbed[lang].vectors[target_id]
            self.lExtracted[lang] = extraction
        return self.lExtracted

    def get_lEmbeddings(self):
        return {lang: self.lEmbed[lang].vectors for lang in self.langs}


def XdotM(X, M, sif):
    E = X.dot(M)
    if sif:
        E = remove_pc(E, npc=1)
    return E


def wce_matrix(X, Y):
    wce = supervised_embeddings_tfidf(X, Y)
    wce = zscores(wce, axis=0)
    return wce


def supervised_embeddings_tfidf(X, Y):
    tfidf_norm = X.sum(axis=0)
    tfidf_norm[tfidf_norm == 0] = 1
    F = (X.T).dot(Y) / tfidf_norm.T
    return F


def zscores(X, axis=0):
    """
    scipy.stats.zscores does not avoid division by 0, which can indeed occur
    :param X:
    :param axis:
    :return:
    """
    std = np.clip(np.std(X, ddof=1, axis=axis), 1e-5, None)
    mean = np.mean(X, axis=axis)
    return (X - mean) / std


