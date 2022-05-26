import pickle

import numpy as np
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from src.util.embeddings_manager import supervised_embeddings_tfidf


class TfidfVectorizerMultilingual:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, lX, ly=None):
        self.langs = sorted(lX.keys())
        self.vectorizer = {l: TfidfVectorizer(**self.kwargs).fit(lX[l]) for l in self.langs}
        return self

    def transform(self, lX):
        return {l: self.vectorizer[l].transform(lX[l]) for l in self.langs}

    def fit_transform(self, lX, ly=None):
        return self.fit(lX, ly).transform(lX)

    def vocabulary(self, l=None):
        if l is None:
            return {l: self.vectorizer[l].vocabulary_ for l in self.langs}
        else:
            return self.vectorizer[l].vocabulary_

    def get_analyzer(self, l=None):
        if l is None:
            return {l: self.vectorizer[l].build_analyzer() for l in self.langs}
        else:
            return self.vectorizer[l].build_analyzer()


def _normalize(lX, l2=True):
    return {lang: normalize(X) for lang, X in lX.items()} if l2 else lX


def none_dict(langs):
    return {l: None for l in langs}


class MultilingualIndex:
    def __init__(self):
        """
        Class that contains monolingual Indexes
        """
        self.l_index = {}
        self.l_vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)

    def index(self, l_devel_raw, l_devel_target, l_test_raw, l_test_target, l_pretrained_vocabulary=None):
        self.langs = sorted(l_devel_raw.keys())
        self.l_vectorizer.fit(l_devel_raw)
        l_vocabulary = self.l_vectorizer.vocabulary()
        l_analyzer = self.l_vectorizer.get_analyzer()
        if l_pretrained_vocabulary is None:
            l_pretrained_vocabulary = none_dict(self.langs)

        for lang in self.langs:
            # Init monolingual Index
            self.l_index[lang] = Index(l_devel_raw[lang], l_devel_target[lang], l_test_raw[lang], l_test_target[lang],
                                       lang)
            # call to index() function of monolingual Index
            self.l_index[lang].index(l_pretrained_vocabulary[lang], l_analyzer[lang], l_vocabulary[lang])

    def train_val_split(self, val_prop=0.2, max_val=2000, seed=42):
        for l, index in self.l_index.items():
            index.train_val_split(val_prop, max_val, seed=seed)

    def embedding_matrices(self, lpretrained, supervised):
        """
        Extract from pretrained embeddings words that are found in the training dataset, then for each language
        calls the respective monolingual index and build the embedding matrix (if supervised, WCE are concatenated
        to the unsupervised vectors).
        :param lpretrained: dict {lang : matrix of word-embeddings }
        :param supervised: bool, whether to deploy Word-Class Embeddings or not
        :return: self
        """
        lXtr = self.get_lXtr() if supervised else none_dict(self.langs)
        lYtr = self.l_train_target() if supervised else none_dict(self.langs)
        lWordList = self.get_wordlist()
        lExtracted = lpretrained.extract(lWordList)
        for lang, index in self.l_index.items():
            # if supervised concatenate embedding matrices of pretrained unsupervised
            # and supervised word-class embeddings
            index.compose_embedding_matrix(lExtracted[lang], supervised, lXtr[lang], lYtr[lang])
            self.sup_range = index.wce_range
        return self

    def get_wordlist(self):
        wordlist = {}
        for lang, index in self.l_index.items():
            wordlist[lang] = index.get_word_list()
        return wordlist

    def get_raw_lXtr(self):
        lXtr_raw = {k: [] for k in self.langs}
        lYtr_raw = {k: [] for k in self.langs}
        for lang in self.langs:
            lXtr_raw[lang] = self.l_index[lang].train_raw
            lYtr_raw[lang] = self.l_index[lang].train_raw
        return lXtr_raw

    def get_raw_lXva(self):
        lXva_raw = {k: [] for k in self.langs}
        for lang in self.langs:
            lXva_raw[lang] = self.l_index[lang].val_raw

        return lXva_raw

    def get_raw_lXte(self):
        lXte_raw = {k: [] for k in self.langs}
        for lang in self.langs:
            lXte_raw[lang] = self.l_index[lang].test_raw

        return lXte_raw

    def get_lXtr(self):
        if not hasattr(self, 'lXtr'):
            self.lXtr = self.l_vectorizer.transform({l: index.train_raw for l, index in self.l_index.items()})
        return self.lXtr

    def get_lXva(self):
        if not hasattr(self, 'lXva'):
            self.lXva = self.l_vectorizer.transform({l: index.val_raw for l, index in self.l_index.items()})
        return self.lXva

    def get_lXte(self):
        if not hasattr(self, 'lXte'):
            self.lXte = self.l_vectorizer.transform({l: index.test_raw for l, index in self.l_index.items()})
        return self.lXte

    def get_target_dim(self):
        return self.l_index[self.langs[0]].devel_target.shape[1]

    def l_vocabsize(self):
        return {l: index.vocabsize for l, index in self.l_index.items()}

    def l_embeddings(self):
        return {l: index.embedding_matrix for l, index in self.l_index.items()}

    def l_pad(self):
        return {l: index.pad_index for l, index in self.l_index.items()}

    def l_train_index(self):
        return {l: index.train_index for l, index in self.l_index.items()}

    def l_train_raw_index(self):
        return {l: index.train_raw for l, index in self.l_index.items()}

    def l_train_target(self):
        return {l: index.train_target for l, index in self.l_index.items()}

    def l_val_index(self):
        return {l: index.val_index for l, index in self.l_index.items()}

    def l_val_raw_index(self):
        return {l: index.val_raw for l, index in self.l_index.items()}

    def l_test_raw_index(self):
        return {l: index.test_raw for l, index in self.l_index.items()}

    def l_devel_raw_index(self):
        return {l: index.devel_raw for l, index in self.l_index.items()}

    def l_val_target(self):
        return {l: index.val_target for l, index in self.l_index.items()}

    def l_test_target(self):
        return {l: index.test_target for l, index in self.l_index.items()}

    def l_test_index(self):
        return {l: index.test_index for l, index in self.l_index.items()}

    def l_devel_index(self):
        return {l: index.devel_index for l, index in self.l_index.items()}

    def l_devel_target(self):
        return {l: index.devel_target for l, index in self.l_index.items()}

    def l_train(self):
        return self.l_train_index(), self.l_train_target()

    def l_val(self):
        return self.l_val_index(), self.l_val_target()

    def l_test(self):
        return self.l_test_index(), self.l_test_target()

    def l_train_raw(self):
        return self.l_train_raw_index(), self.l_train_target()

    def l_val_raw(self):
        return self.l_val_raw_index(), self.l_val_target()

    def l_test_raw(self):
        return self.l_test_raw_index(), self.l_test_target()

    def l_devel_raw(self):
        return self.l_devel_raw_index(), self.l_devel_target()

    def get_l_pad_index(self):
        return {l: index.get_pad_index() for l, index in self.l_index.items()}


class Index:
    def __init__(self, devel_raw, devel_target, test_raw, test_target, lang):
        """
        Monolingual Index, takes care of tokenizing raw data, converting strings to ids, splitting the data into
        training and validation.
        :param devel_raw: list of strings, list of raw training texts
        :param devel_target:
        :param test_raw: list of strings, list of raw test texts
        :param lang: list, list of languages contained in the dataset
        """
        self.lang = lang
        self.devel_raw = devel_raw
        self.devel_target = devel_target
        self.test_raw = test_raw
        self.test_target = test_target

    def index(self, pretrained_vocabulary, analyzer, vocabulary):
        self.word2index = dict(vocabulary)
        known_words = set(self.word2index.keys())
        if pretrained_vocabulary is not None:
            known_words.update(pretrained_vocabulary)

        self.word2index['UNKTOKEN'] = len(self.word2index)
        self.word2index['PADTOKEN'] = len(self.word2index)
        self.unk_index = self.word2index['UNKTOKEN']
        self.pad_index = self.word2index['PADTOKEN']

        # index documents and keep track of test terms outside the development vocabulary that are in Muse (if available)
        self.out_of_vocabulary = dict()
        self.devel_index = index(self.devel_raw, self.word2index, known_words, analyzer, self.unk_index,
                                 self.out_of_vocabulary)
        self.test_index = index(self.test_raw, self.word2index, known_words, analyzer, self.unk_index,
                                self.out_of_vocabulary)

        self.vocabsize = len(self.word2index) + len(self.out_of_vocabulary)

        print(f'[indexing complete for lang {self.lang}] vocabulary-size={self.vocabsize}')

    def get_pad_index(self):
        return self.pad_index

    def train_val_split(self, val_prop, max_val, seed):
        devel = self.devel_index
        target = self.devel_target
        devel_raw = self.devel_raw

        val_size = int(min(len(devel) * val_prop, max_val))

        self.train_index, self.val_index, self.train_target, self.val_target, self.train_raw, self.val_raw = \
            train_test_split(
                devel, target, devel_raw, test_size=val_size, random_state=seed, shuffle=True)

        print(
            f'split lang {self.lang}: train={len(self.train_index)} val={len(self.val_index)} test={len(self.test_index)}')

    def get_word_list(self):
        def extract_word_list(word2index):
            return [w for w, i in sorted(word2index.items(), key=lambda x: x[1])]

        word_list = extract_word_list(self.word2index)
        word_list += extract_word_list(self.out_of_vocabulary)
        return word_list

    def compose_embedding_matrix(self, pretrained, supervised, Xtr=None, Ytr=None):
        print(f'[generating embedding matrix for lang {self.lang}]')

        self.wce_range = None
        embedding_parts = []

        if pretrained is not None:
            print('\t[pretrained-matrix]')
            embedding_parts.append(pretrained)
            del pretrained

        if supervised:
            print('\t[supervised-matrix]')
            F = supervised_embeddings_tfidf(Xtr, Ytr)
            num_missing_rows = self.vocabsize - F.shape[0]
            F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()

            offset = 0
            if embedding_parts:
                offset = embedding_parts[0].shape[1]
            self.wce_range = [offset, offset + F.shape[1]]
            embedding_parts.append(F)

        self.embedding_matrix = torch.cat(embedding_parts, dim=1)

        print(f'[embedding matrix for lang {self.lang} has shape {self.embedding_matrix.shape}]')


def index(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary):
    """
    Index (i.e., replaces word strings with numerical indexes) a list of string documents
    :param data: list of string documents
    :param vocab: a fixed mapping [str]->[int] of words to indexes
    :param known_words: a set of known words (e.g., words that, despite not being included in the vocab, can be retained
    because they are anyway contained in a pre-trained embedding set that we know in advance)
    :param analyzer: the preprocessor in charge of transforming the document string into a chain of string words
    :param unk_index: the index of the 'unknown token', i.e., a symbol that characterizes all words that we cannot keep
    :param out_of_vocabulary: an incremental mapping [str]->[int] of words to indexes that will index all those words that
    are not in the original vocab but that are in the known_words
    :return:
    """
    indexes = []
    vocabsize = len(vocab)
    unk_count = 0
    knw_count = 0
    out_count = 0
    # pbar = tqdm(data, desc=f'indexing')
    for text in data:
        words = analyzer(text)
        index = []
        for word in words:
            if word in vocab:
                idx = vocab[word]
            else:
                if word in known_words:
                    if word not in out_of_vocabulary:
                        out_of_vocabulary[word] = vocabsize + len(out_of_vocabulary)
                    idx = out_of_vocabulary[word]
                    out_count += 1
                else:
                    idx = unk_index
                    unk_count += 1
            index.append(idx)
        indexes.append(index)
        knw_count += len(index)
        # pbar.set_description(f'[unk = {unk_count}/{knw_count}={(100.*unk_count/knw_count):.2f}%]'
        #                      f'[out = {out_count}/{knw_count}={(100.*out_count/knw_count):.2f}%]')
    return indexes


def is_true(tensor, device):
    return torch.where(tensor == 1, torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))


def is_false(tensor, device):
    return torch.where(tensor == 0, torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))


def define_pad_length(index_list):
    lengths = [len(index) for index in index_list]
    return int(np.mean(lengths) + np.std(lengths))


def pad(index_list, pad_index, max_pad_length=None):
    pad_length = np.max([len(index) for index in index_list])
    if max_pad_length is not None:
        pad_length = min(pad_length, max_pad_length)
    for i, indexes in enumerate(index_list):
        index_list[i] = [pad_index] * (pad_length - len(indexes)) + indexes[:pad_length]
    return index_list


def get_params(optimc=False):
    if not optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    kernel = 'rbf'
    return [{'kernel': [kernel], 'C': c_range, 'gamma':['auto']}]


def get_method_name(args):
    _id = ''
    _id_conf = [args.post_embedder, args.wce_embedder, args.muse_embedder, args.bert_embedder, args.gru_embedder]
    _id_name = ['X', 'W', 'M', 'B', 'G']
    for i, conf in enumerate(_id_conf):
        if conf:
            _id += _id_name[i]
    _id = _id if not args.rnn_wce else _id + '_wce'
    _dataset_path = args.dataset.split('/')[-1].split('_')
    dataset_id = _dataset_path[0] + _dataset_path[-1]
    return _id, dataset_id


def reduce_docs(docs: list, n=250):
    for d in docs:
        for k, v in d.items():
            d[k] = v[:n]
    return docs


def dump_predictions(preds: dict, true: dict):
    with open(f"result_dumps/res_rcvrun0.pkl", "wb") as f:
        pickle.dump((preds, true), f)
