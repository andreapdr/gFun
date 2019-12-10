import os
import pickle
import numpy as np
from torchtext.vocab import Vectors
import torch
from abc import ABC, abstractmethod
from data.supervised import get_supervised_embeddings


class PretrainedEmbeddings(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def vocabulary(self): pass

    @abstractmethod
    def dim(self): pass

    @classmethod
    def reindex(cls, words, word2index):
        source_idx, target_idx = [], []
        for i, word in enumerate(words):
            if word not in word2index: continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        return source_idx, target_idx


class WordEmbeddings:

    def __init__(self, lang, we, worddim):
        self.lang = lang
        self.we = we
        self.worddim = worddim
        self.dimword = {v:k for k,v in self.worddim.items()}

    @classmethod
    def load(cls, basedir, lang, word_preprocessor=None, dopickle=True):
        filename = 'wiki.multi.{}.vec'.format(lang)
        we_path = os.path.join(basedir, filename)

        if dopickle and os.path.exists(we_path + '.pkl'):
            print('loading pkl in {}'.format(we_path + '.pkl'))
            (worddim, we) = pickle.load(open(we_path + '.pkl', 'rb'))
        else:
            word_registry=set()
            lines = open(we_path).readlines()
            nwords, dims = [int(x) for x in lines[0].split()]
            print('reading we of {} dimensions'.format(dims))
            we = np.zeros((nwords, dims), dtype=float)
            worddim = {}
            index = 0
            for i, line in enumerate(lines[1:]):
                if (i + 1) % 100 == 0:
                    print('\r{}/{}'.format(i + 1, len(lines)), end='')
                word, *vals = line.split()
                wordp = word_preprocessor(word) if word_preprocessor is not None else word
                if wordp:
                    wordp=wordp[0]
                    if wordp in word_registry:
                        print('warning: word <{}> generates a duplicate <{}> after preprocessing'.format(word,wordp))
                    elif len(vals) == dims:
                        worddim[wordp] = index
                        we[index, :] = np.array(vals).astype(float)
                        index+=1
                # else:
                #     print('warning: word <{}> generates an empty string after preprocessing'.format(word))
            we = we[:index]
            print('load {} words'.format(index))
            if dopickle:
                print('saving...')
                pickle.dump((worddim, we), open(we_path + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

        return WordEmbeddings(lang, we, worddim)

    def vocabulary(self):
        return set(self.worddim.keys())

    def __getitem__(self, key):
        return self.we[self.worddim[key]]

    def dim(self):
        return self.we.shape[1]

    def __contains__(self, key):
        return key in self.worddim

    def most_similar(self, word_vect, k):
        if word_vect.ndim == 1:
            word_vect = word_vect.reshape(1,-1)
        assert word_vect.shape[1] == self.dim(), 'inconsistent dimensions'

        sim = np.dot(word_vect,self.we.T)
        order = np.argsort(-1*sim, axis=1)[:,:k]

        similar_words = [[self.dimword[order[vi,ki]] for ki in range(k)] for vi in range(word_vect.shape[0])]
        sim_scores = sim[:,order]
        return similar_words, sim_scores

    def get_vectors(self, wordlist):
        indexes = np.array([self.worddim[w] for w in wordlist])
        return self.we[indexes]

    def restrict(self, vocabulary):
        # vocabulary is a set of terms to be kept
        active_vocabulary = sorted([w for w in vocabulary if w in self.worddim])
        lost = len(vocabulary)-len(active_vocabulary)
        if lost>0: #some termr are missing, so it will be replaced by UNK
            print('warning: missing {} terms for lang {}'.format(lost, self.lang))
        self.we = self.get_vectors(active_vocabulary)
        assert self.we.shape[0]==len(active_vocabulary)
        self.dimword={i:w for i,w in enumerate(active_vocabulary)}
        self.worddim={w:i for i,w in enumerate(active_vocabulary)}
        return self

    @classmethod
    def load_poly(cls, basedir, langs, lang_vocabularies=None, word_preprocessor=None):
        if lang_vocabularies is None:
            return cls.merge([cls.load(basedir,lang, word_preprocessor) for lang in langs])
        else:
            # assert all([l in lang_vocabularies for l in langs]), 'missing vocabulary for some languages'
            return cls.merge([cls.load(basedir, lang, word_preprocessor).restrict(lang_vocabularies[lang]) for lang in langs])

    @classmethod
    def merge(cls, we_list):
        assert all([isinstance(we, WordEmbeddings) for we in we_list]), \
            'instances of {} expected'.format(WordEmbeddings.__name__)

        polywe = []
        worddim={}
        offset=0
        for we in we_list:
            polywe.append(we.we)
            worddim.update({'{}::{}'.format(we.lang, w):d+offset for w,d in we.worddim.items()})
            offset=len(worddim)
        polywe = np.vstack(polywe)

        return WordEmbeddings(lang='poly', we=polywe, worddim=worddim)


class FastTextWikiNews(Vectors):

    url_base = 'Cant auto-download MUSE embeddings'
    path = '/storage/andrea/FUNNELING/embeddings/wiki.multi.{}.vec'
    _name = 'wiki.multi.{}.vec'

    def __init__(self, cache, language="en", **kwargs):
        url = self.url_base.format(language)
        # name = self.path.format(language)
        name = cache + self._name.format(language)
        # print(f'\n\nFASTEXTWIKI-NEW CLASS:\nurl = {url}\nname = {name}\ncache {cache}\nlanguage = {language}')
        super(FastTextWikiNews, self).__init__(name, cache=cache, url=url, **kwargs)


class FastTextMUSE(PretrainedEmbeddings):

    def __init__(self, path, lang, limit=None):
        super().__init__()
        print(f'Loading fastText pretrained vectors from {path}')
        assert os.path.exists(path), print(f'pre-trained vectors not found in {path}')
        self.embed = FastTextWikiNews(path, lang, max_vectors=limit)
        # print('Done')

    def vocabulary(self):
        return set(self.embed.stoi.keys())

    def dim(self):
        return self.embed.dim

    def extract(self, words):
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.embed.stoi)
        extraction = torch.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        return extraction


def embedding_matrix(path, voc, lang):
    vocabulary = np.asarray(list(zip(*sorted(voc.items(), key=lambda x:x[1])))[0])

    print('[embedding matrix]')
    print(f'# [pretrained-matrix: FastTextMUSE {lang}]')
    pretrained = FastTextMUSE(path, lang)
    P = pretrained.extract(vocabulary).numpy()
    del pretrained
    print(f'[embedding matrix done] of shape={P.shape}\n')

    return vocabulary, P


def WCE_matrix(Xtr, Ytr, lang):
    print('\n# [supervised-matrix]')
    S = get_supervised_embeddings(Xtr[lang], Ytr[lang], max_label_space=50)
    print(f'[embedding matrix done] of shape={S.shape}\n')
    return S
