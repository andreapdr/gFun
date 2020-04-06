import warnings
import time
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from embeddings.supervised import get_supervised_embeddings
from learning.transformers import PosteriorProbabilitiesEmbedder, TfidfVectorizerMultilingual
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from tqdm import tqdm
import torch
from scipy.sparse import vstack, issparse


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
    indexes=[]
    vocabsize = len(vocab)
    unk_count = 0
    knw_count = 0
    out_count = 0
    pbar = tqdm(data, desc=f'indexing documents')
    for text in pbar:
        words = analyzer(text)
        index = []
        for word in words:
            if word in vocab:
                idx = vocab[word]
            else:
                if word in known_words:
                    if word not in out_of_vocabulary:
                        out_of_vocabulary[word] = vocabsize+len(out_of_vocabulary)
                    idx = out_of_vocabulary[word]
                    out_count += 1
                else:
                    idx = unk_index
                    unk_count += 1
            index.append(idx)
        indexes.append(index)
        knw_count += len(index)
        pbar.set_description(f'[unk = {unk_count}/{knw_count}={(100.*unk_count/knw_count):.2f}%]'
                             f'[out = {out_count}/{knw_count}={(100.*out_count/knw_count):.2f}%]')
    return indexes


def define_pad_length(index_list):
    lengths = [len(index) for index in index_list]
    return int(np.mean(lengths)+np.std(lengths))


def pad(index_list, pad_index, max_pad_length=None):
    pad_length = np.max([len(index) for index in index_list])
    if max_pad_length is not None:
        pad_length = min(pad_length, max_pad_length)
    for i,indexes in enumerate(index_list):
        index_list[i] = [pad_index]*(pad_length-len(indexes)) + indexes[:pad_length]
    return index_list


class Index:
    def __init__(self, devel_raw, devel_target, test_raw, lang):
        self.lang = lang
        self.devel_raw = devel_raw
        self.devel_target = devel_target
        self.test_raw = test_raw

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
        self.devel_index = index(self.devel_raw, self.word2index, known_words, analyzer, self.unk_index, self.out_of_vocabulary)
        self.test_index = index(self.test_raw, self.word2index, known_words, analyzer, self.unk_index, self.out_of_vocabulary)

        self.vocabsize = len(self.word2index) + len(self.out_of_vocabulary)

        print(f'[indexing complete for lang {self.lang}] vocabulary-size={self.vocabsize}')

    def train_val_split(self, val_prop, max_val, seed):
        devel = self.devel_index
        target = self.devel_target
        devel_raw = self.devel_raw

        val_size = int(min(len(devel) * val_prop, max_val))

        self.train_index, self.val_index, self.train_target, self.val_target, self.train_raw, self.val_raw = \
            train_test_split(
                devel, target, devel_raw, test_size=val_size, random_state=seed, shuffle=True
            )

        print(f'split lang {self.lang}: train={len(self.train_index)} val={len(self.val_index)} test={len(self.test_index)}')

    def get_word_list(self):
        def extract_word_list(word2index):
            return [w for w,i in sorted(word2index.items(), key=lambda x: x[1])]

        word_list = extract_word_list(self.word2index)
        word_list += extract_word_list(self.out_of_vocabulary)
        return word_list

    def compose_embedding_matrix(self, pretrained, supervised, Xtr=None, Ytr=None):
        print(f'[generating embedding matrix for lang {self.lang}]')

        self.wce_range = None
        embedding_parts = []

        if pretrained is not None:
            print('\t[pretrained-matrix]')
            word_list = self.get_word_list()
            muse_embeddings = pretrained.extract(word_list)
            embedding_parts.append(muse_embeddings)
            del pretrained

        if supervised:
            print('\t[supervised-matrix]')
            F = get_supervised_embeddings(Xtr, Ytr, reduction=None, method='dotn')
            num_missing_rows = self.vocabsize - F.shape[0]
            F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()

            offset = 0
            if embedding_parts:
                offset = embedding_parts[0].shape[1]
            self.wce_range = [offset, offset + F.shape[1]]

            embedding_parts.append(F)

        make_dumps = False
        if make_dumps:
            print(f'Dumping Embedding Matrices ...')
            import pickle
            with open(f'../dumps/dump_{self.lang}_rcv.pkl', 'wb') as outfile:
                pickle.dump((self.lang, embedding_parts, self.word2index), outfile)
            with open(f'../dumps/corpus_{self.lang}_rcv.pkl', 'wb') as outfile2:
                pickle.dump((self.lang, self.devel_raw, self.devel_target), outfile2)

        self.embedding_matrix = torch.cat(embedding_parts, dim=1)

        print(f'[embedding matrix for lang {self.lang} has shape {self.embedding_matrix.shape}]')


def none_dict(langs):
    return {l:None for l in langs}

class MultilingualIndex:
    def __init__(self): #, add_language_trace=False):
        self.l_index = {}
        # self.l_vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True)
        self.l_vectorizer = TfidfVectorizerMultilingual(sublinear_tf=True, use_idf=True, max_features=25000)
        # self.add_language_trace=add_language_trace

    def index(self, l_devel_raw, l_devel_target, l_test_raw, l_pretrained_vocabulary):
        self.langs = sorted(l_devel_raw.keys())

        #build the vocabularies
        self.l_vectorizer.fit(l_devel_raw)
        l_vocabulary = self.l_vectorizer.vocabulary()
        l_analyzer = self.l_vectorizer.get_analyzer()

        for l in self.langs:
            self.l_index[l] = Index(l_devel_raw[l], l_devel_target[l], l_test_raw[l], l)
            self.l_index[l].index(l_pretrained_vocabulary[l], l_analyzer[l], l_vocabulary[l])

    def train_val_split(self, val_prop=0.2, max_val=2000, seed=42):
        for l,index in self.l_index.items():
            index.train_val_split(val_prop, max_val, seed=seed)

    def embedding_matrices(self, lpretrained, supervised):
        lXtr = self.get_lXtr() if supervised else none_dict(self.langs)
        lYtr = self.l_train_target() if supervised else none_dict(self.langs)
        for l,index in self.l_index.items():
            index.compose_embedding_matrix(lpretrained[l], supervised, lXtr[l], lYtr[l])
            self.sup_range = index.wce_range

            # experimental... does it make sense to keep track of the language? i.e., to inform the network from which
            # language does the data came from...
            # if self.add_language_trace and pretrained_embeddings is not None:
            #     print('adding language trace')
            #     lang_trace = torch.zeros(size=(vocabsize, len(self.langs)))
            #     lang_trace[:,i]=1
            #     pretrained_embeddings = torch.cat([pretrained_embeddings, lang_trace], dim=1)


    def posterior_probabilities(self, max_training_docs_by_lang=5000, store_posteriors=False, stored_post=False):
        # choose a maximum of "max_training_docs_by_lang" for training the calibrated SVMs
        timeit = time.time()
        lXtr = {l:Xtr for l,Xtr in self.get_lXtr().items()}
        lYtr = {l:Ytr for l,Ytr in self.l_train_target().items()}
        if not stored_post:
            for l in self.langs:
                n_elements = lXtr[l].shape[0]
                if n_elements > max_training_docs_by_lang:
                    choice = np.random.permutation(n_elements)[:max_training_docs_by_lang]
                    lXtr[l] = lXtr[l][choice]
                    lYtr[l] = lYtr[l][choice]

            # train the posterior probabilities embedder
            print('[posteriors] training a calibrated SVM')
            learner = SVC(kernel='linear', probability=True, cache_size=1000, C=1, random_state=1, gamma='auto')
            prob_embedder = PosteriorProbabilitiesEmbedder(learner, l2=False)
            prob_embedder.fit(lXtr, lYtr)

            # transforms the training, validation, and test sets into posterior probabilities
            print('[posteriors] generating posterior probabilities')
            lPtr = prob_embedder.transform(self.get_lXtr())
            lPva = prob_embedder.transform(self.get_lXva())
            lPte = prob_embedder.transform(self.get_lXte())
        # NB: Check splits indices !
            if store_posteriors:
                import pickle
                with open('../dumps/posteriors_fulljrc.pkl', 'wb') as outfile:
                    pickle.dump([lPtr, lPva, lPte], outfile)
                    print(f'Successfully dumped posteriors!')
        else:
            import pickle
            with open('../dumps/posteriors_fulljrc.pkl', 'rb') as infile:
                lPtr, lPva, lPte = pickle.load(infile)
                print(f'Successfully loaded stored posteriors!')
        print(f'[posteriors] done in {time.time() - timeit}')
        return lPtr, lPva, lPte

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

    def l_vocabsize(self):
        return {l:index.vocabsize for l,index in self.l_index.items()}

    def l_embeddings(self):
        return {l:index.embedding_matrix for l,index in self.l_index.items()}

    def l_pad(self):
        return {l: index.pad_index for l, index in self.l_index.items()}

    def l_train_index(self):
        return {l: index.train_index for l, index in self.l_index.items()}

    def l_train_target(self):
        return {l: index.train_target for l, index in self.l_index.items()}

    def l_val_index(self):
        return {l: index.val_index for l, index in self.l_index.items()}

    def l_val_target(self):
        return {l: index.val_target for l, index in self.l_index.items()}

    def l_test_index(self):
        return {l: index.test_index for l, index in self.l_index.items()}

    def l_train(self):
        return self.l_train_index(), self.l_train_target()

    def l_val(self):
        return self.l_val_index(), self.l_val_target()



class Batch:
    def __init__(self, batchsize, batches_per_epoch, languages, lpad, max_pad_length=500):
        self.batchsize = batchsize
        self.batches_per_epoch = batches_per_epoch
        self.languages = languages
        self.lpad=lpad
        self.max_pad_length=max_pad_length
        self.init_offset()

    def init_offset(self):
        self.offset = {lang: 0 for lang in self.languages}

    def batchify(self, l_index, l_post, llabels):
        langs = self.languages
        l_num_samples = {l:len(l_index[l]) for l in langs}

        max_samples = max(l_num_samples.values())
        n_batches = max_samples // self.batchsize + 1 * (max_samples % self.batchsize > 0)
        if self.batches_per_epoch != -1 and self.batches_per_epoch < n_batches:
            n_batches = self.batches_per_epoch

        for b in range(n_batches):
            for lang in langs:
                index, labels = l_index[lang], llabels[lang]
                offset = self.offset[lang]
                if offset >= l_num_samples[lang]:
                    offset = 0
                limit = offset+self.batchsize

                batch_slice = slice(offset, limit)
                batch = index[batch_slice]
                batch_labels = labels[batch_slice].toarray()

                post = None
                if l_post is not None:
                    post = torch.FloatTensor(l_post[lang][batch_slice]).cuda()

                batch = pad(batch, pad_index=self.lpad[lang], max_pad_length=self.max_pad_length)

                batch = torch.LongTensor(batch).cuda()
                target = torch.FloatTensor(batch_labels).cuda()

                self.offset[lang] = limit

                yield batch, post, target, lang


def batchify(l_index, l_post, llabels, batchsize, lpad, max_pad_length=500):
    langs = sorted(l_index.keys())
    nsamples = max([len(l_index[l]) for l in langs])
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    for b in range(nbatches):
        for lang in langs:
            index, labels = l_index[lang], llabels[lang]

            if b * batchsize >= len(index):
                continue
            batch = index[b*batchsize:(b+1)*batchsize]
            batch_labels = labels[b*batchsize:(b+1)*batchsize].toarray()
            post = None
            if l_post is not None:
                post = torch.FloatTensor(l_post[lang][b*batchsize:(b+1)*batchsize]).cuda()
            batch = pad(batch, pad_index=lpad[lang], max_pad_length=max_pad_length)
            batch = torch.LongTensor(batch)
            target = torch.FloatTensor(batch_labels)
            yield batch.cuda(), post, target.cuda(), lang


def batchify_unlabelled(index_list, batchsize, pad_index, max_pad_length=500):
    nsamples = len(index_list)
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    for b in range(nbatches):
        batch = index_list[b*batchsize:(b+1)*batchsize]
        batch = pad(batch, pad_index=pad_index, max_pad_length=max_pad_length)
        batch = torch.LongTensor(batch)
        yield batch.cuda()


def clip_gradient(model, clip_value=1e-1):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def predict(logits, classification_type='multilabel'):
    if classification_type == 'multilabel':
        prediction = torch.sigmoid(logits) > 0.5
    elif classification_type == 'singlelabel':
        prediction = torch.argmax(logits, dim=1).view(-1, 1)
    else:
        print('unknown classification type')

    return prediction.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






