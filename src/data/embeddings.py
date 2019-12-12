import os
import pickle
from torchtext.vocab import Vectors
import torch
from abc import ABC, abstractmethod
from data.supervised import get_supervised_embeddings
from util.decompositions import *


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
            word_registry = set()
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
                    wordp = wordp[0]
                    if wordp in word_registry:
                        print('warning: word <{}> generates a duplicate <{}> after preprocessing'.format(word,wordp))
                    elif len(vals) == dims:
                        worddim[wordp] = index
                        we[index, :] = np.array(vals).astype(float)
                        index += 1
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
        if lost > 0:    # some terms are missing, so it will be replaced by UNK
            print('warning: missing {} terms for lang {}'.format(lost, self.lang))
        self.we = self.get_vectors(active_vocabulary)
        assert self.we.shape[0] == len(active_vocabulary)
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
        worddim = {}
        offset = 0
        for we in we_list:
            polywe.append(we.we)
            worddim.update({'{}::{}'.format(we.lang, w):d+offset for w,d in we.worddim.items()})
            offset = len(worddim)
        polywe = np.vstack(polywe)

        return WordEmbeddings(lang='poly', we=polywe, worddim=worddim)


class FastTextWikiNews(Vectors):

    url_base = 'Cant auto-download MUSE embeddings'
    path = '/storage/andrea/FUNNELING/embeddings/wiki.multi.{}.vec'
    _name = '/embeddings/wiki.multi.{}.vec'

    def __init__(self, cache, language="en", **kwargs):
        url = self.url_base.format(language)
        name = cache + self._name.format(language)
        super(FastTextWikiNews, self).__init__(name, cache=cache, url=url, **kwargs)


class EmbeddingsAligned(Vectors):

    def __init__(self, type, path, lang, voc):
        # todo - rewrite as relative path
        self.name = '/embeddings/wiki.multi.{}.vec' if type == 'MUSE' else '/embeddings_polyFASTTEXT/wiki.{}.align.vec'
        self.cache_path = '/home/andreapdr/CLESA/embeddings' if type == 'MUSE' else '/home/andreapdr/CLESA/embeddings_polyFASTTEXT'
        self.path = path + self.name.format(lang)
        assert os.path.exists(path), f'pre-trained vectors not found in {path}'
        super(EmbeddingsAligned, self).__init__(self.path, cache=self.cache_path)
        self.vectors = self.extract(voc)

    def vocabulary(self):
        return set(self.stoi.keys())

    def extract(self, words):
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.stoi)
        extraction = torch.zeros((len(words), self.dim))
        extraction[source_idx] = self.vectors[target_idx]
        return extraction

    def reduce(self, dim):
        pca = PCA(n_components=dim)
        self.vectors = pca.fit_transform(self.vectors)
        return


class FastTextMUSE(PretrainedEmbeddings):

    def __init__(self, path, lang, limit=None):
        super().__init__()
        print(f'Loading fastText pretrained vectors from {path}')
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


class StorageEmbeddings:
    def __init__(self, path):
        self.path = path
        self.lang_U = dict()
        self.lang_S = dict()

    def _add_embeddings_unsupervised(self, type, docs, vocs, max_label_space=300):
        for lang in docs.keys():
            print(f'# [unsupervised-matrix {type}] for {lang}')
            voc = np.asarray(list(zip(*sorted(vocs[lang].items(), key=lambda x: x[1])))[0])
            self.lang_U[lang] = EmbeddingsAligned(type, self.path, lang, voc).vectors
            print(f'Matrix U (weighted sum) of shape {self.lang_U[lang].shape}\n')
            nC = self.lang_U[lang].shape[1]
        if max_label_space == 0:
            print(f'Computing optimal number of PCA components along matrices U')
            optimal_n = get_optimal_dim(self.lang_U, 'U')
            self.lang_U = run_pca(optimal_n, self.lang_U)
        elif max_label_space < nC:
            self.lang_U = run_pca(max_label_space, self.lang_U)


        return

    def _add_emebeddings_supervised(self, docs, labels, reduction, max_label_space, voc):
        for lang in docs.keys():    # compute supervised matrices S - then apply PCA
            print(f'# [supervised-matrix] for {lang}')
            self.lang_S[lang] = get_supervised_embeddings(docs[lang], labels[lang], reduction, max_label_space, voc[lang], lang)
            nC = self.lang_S[lang].shape[1]
            print(f'[embedding matrix done] of shape={self.lang_S[lang].shape}\n')

        if max_label_space == 0:
            print(f'Computing optimal number of PCA components along matrices S')
            optimal_n = get_optimal_dim(self.lang_S, 'S')
            self.lang_S = run_pca(optimal_n, self.lang_S)
        elif max_label_space == -1:
            print(f'Computing PCA on vertical stacked WCE embeddings')
            languages = self.lang_S.keys()
            _temp_stack = np.vstack([self.lang_S[lang] for lang in languages])
            stacked_pca = PCA(n_components=50)
            stacked_pca.fit(_temp_stack)
            for lang in languages:
                self.lang_S[lang] = stacked_pca.transform(self.lang_S[lang])
        elif max_label_space < nC:
            self.lang_S = run_pca(max_label_space, self.lang_S)

        return

    def _concatenate_embeddings(self, docs):
        _r = dict()
        for lang in self.lang_U.keys():
            _r[lang] = np.hstack((docs[lang].dot(self.lang_U[lang]), docs[lang].dot(self.lang_S[lang])))
        return _r

    def fit(self, config, docs, vocs, labels):
        if config['unsupervised']:
            self._add_embeddings_unsupervised(config['we_type'], docs, vocs, config['dim_reduction_unsupervised'])
        if config['supervised']:
            self._add_emebeddings_supervised(docs, labels, config['reduction'], config['max_label_space'], vocs)
        return self

    def predict(self, config, docs):
        if config['supervised'] and config['unsupervised']:
            return self._concatenate_embeddings(docs)
        elif config['supervised']:
            _r = dict()
            for lang in docs.keys():
                _r[lang] = docs[lang].dot(self.lang_S[lang])
        else:
            _r = dict()
            for lang in docs.keys():
                _r[lang] = docs[lang].dot(self.lang_U[lang])
        return _r
