import itertools
import re
from os.path import exists

import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.data.languages import NLTK_LANGMAP, RCV2_LANGS_WITH_NLTK_STEMMING
from src.data.reader.jrcacquis_reader import *
from src.data.reader.rcv_reader import fetch_RCV1, fetch_RCV2
from src.data.text_preprocessor import NLTKStemTokenizer, preprocess_documents


class MultilingualDataset:
    """
    A multilingual dataset is a dictionary of training and test documents indexed by language code.
    Train and test sets are represented as tuples of the type (X,Y,ids), where X is a matrix representation of the
    documents (e.g., a document-by-term sparse csr_matrix), Y is a document-by-label binary np.array indicating the
    labels of each document, and ids is a list of document-identifiers from the original collection.
    """

    def __init__(self):
        self.dataset_name = ""
        self.multiling_dataset = {}

    def add(self, lang, Xtr, Ytr, Xte, Yte, tr_ids=None, te_ids=None):
        self.multiling_dataset[lang] = ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids))

    def save(self, file):
        self.sort_indexes()
        pickle.dump(self, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)
        return self

    def __getitem__(self, item):
        if item in self.langs():
            return self.multiling_dataset[item]
        return None

    @classmethod
    def load(cls, file):
        data = pickle.load(open(file, 'rb'))
        data.sort_indexes()
        return data

    @classmethod
    def load_ids(cls, file):
        data = pickle.load(open(file, 'rb'))
        tr_ids = {lang:tr_ids for (lang,((_,_,tr_ids), (_,_,_))) in data.multiling_dataset.items()}
        te_ids = {lang: te_ids for (lang, ((_, _, _), (_, _, te_ids))) in data.multiling_dataset.items()}
        return tr_ids, te_ids

    def sort_indexes(self):
        for (lang, ((Xtr,_,_),(Xte,_,_))) in self.multiling_dataset.items():
            if issparse(Xtr): Xtr.sort_indices()
            if issparse(Xte): Xte.sort_indices()

    def set_view(self, categories=None, languages=None):
        if categories is not None:
            if isinstance(categories, int):
                categories = np.array([categories])
            elif isinstance(categories, list):
                categories = np.array(categories)
            self.categories_view = categories
        if languages is not None:
            self.languages_view = languages

    def training(self, mask_numbers=False, target_as_csr=False):
        return self.lXtr(mask_numbers), self.lYtr(as_csr=target_as_csr)

    def test(self, mask_numbers=False, target_as_csr=False):
        return self.lXte(mask_numbers), self.lYte(as_csr=target_as_csr)

    def lXtr(self, mask_numbers=False):
        proc = lambda x:_mask_numbers(x) if mask_numbers else x
        # return {lang: Xtr for (lang, ((Xtr, _, _), _)) in self.multiling_dataset.items() if lang in self.langs()}
        return {lang:proc(Xtr) for (lang, ((Xtr,_,_),_)) in self.multiling_dataset.items() if lang in self.langs()}

    def lXte(self, mask_numbers=False):
        proc = lambda x: _mask_numbers(x) if mask_numbers else x
        # return {lang: Xte for (lang, (_, (Xte, _, _))) in self.multiling_dataset.items() if lang in self.langs()}
        return {lang:proc(Xte) for (lang, (_,(Xte,_,_))) in self.multiling_dataset.items() if lang in self.langs()}

    def lYtr(self, as_csr=False):
        lY = {lang:self.cat_view(Ytr) for (lang, ((_,Ytr,_),_)) in self.multiling_dataset.items() if lang in self.langs()}
        if as_csr:
            lY = {l:csr_matrix(Y) for l,Y in lY.items()}
        return lY

    def lYte(self, as_csr=False):
        lY = {lang:self.cat_view(Yte) for (lang, (_,(_,Yte,_))) in self.multiling_dataset.items() if lang in self.langs()}
        if as_csr:
            lY = {l:csr_matrix(Y) for l,Y in lY.items()}
        return lY

    def cat_view(self, Y):
        if hasattr(self, 'categories_view'):
            return Y[:,self.categories_view]
        else:
            return Y

    def langs(self):
        if hasattr(self, 'languages_view'):
            langs = self.languages_view
        else:
            langs = sorted(self.multiling_dataset.keys())
        return langs

    def num_categories(self):
        return self.lYtr()[self.langs()[0]].shape[1]

    def show_dimensions(self):
        def shape(X):
            return X.shape if hasattr(X, 'shape') else len(X)
        for (lang, ((Xtr, Ytr, IDtr), (Xte, Yte, IDte))) in self.multiling_dataset.items():
            if lang not in self.langs(): continue
            print("Lang {}, Xtr={}, ytr={}, Xte={}, yte={}".format(lang, shape(Xtr), self.cat_view(Ytr).shape, shape(Xte), self.cat_view(Yte).shape))

    def show_category_prevalences(self):
        nC = self.num_categories()
        accum_tr = np.zeros(nC, dtype=np.int)
        accum_te = np.zeros(nC, dtype=np.int)
        in_langs = np.zeros(nC, dtype=np.int)   # count languages with at least one positive example (per category)
        for (lang, ((Xtr, Ytr, IDtr), (Xte, Yte, IDte))) in self.multiling_dataset.items():
            if lang not in self.langs(): continue
            prev_train = np.sum(self.cat_view(Ytr), axis=0)
            prev_test = np.sum(self.cat_view(Yte), axis=0)
            accum_tr += prev_train
            accum_te += prev_test
            in_langs += (prev_train>0)*1
            print(lang+'-train', prev_train)
            print(lang+'-test', prev_test)
        print('all-train', accum_tr)
        print('all-test', accum_te)

        return accum_tr, accum_te, in_langs

    def set_labels(self, labels):
        self.labels = labels

def _mask_numbers(data):
    mask_moredigit = re.compile(r'\s[\+-]?\d{5,}([\.,]\d*)*\b')
    mask_4digit = re.compile(r'\s[\+-]?\d{4}([\.,]\d*)*\b')
    mask_3digit = re.compile(r'\s[\+-]?\d{3}([\.,]\d*)*\b')
    mask_2digit = re.compile(r'\s[\+-]?\d{2}([\.,]\d*)*\b')
    mask_1digit = re.compile(r'\s[\+-]?\d{1}([\.,]\d*)*\b')
    masked = []
    for text in tqdm(data, desc='masking numbers'):
        text = ' ' + text
        text = mask_moredigit.sub(' MoreDigitMask', text)
        text = mask_4digit.sub(' FourDigitMask', text)
        text = mask_3digit.sub(' ThreeDigitMask', text)
        text = mask_2digit.sub(' TwoDigitMask', text)
        text = mask_1digit.sub(' OneDigitMask', text)
        masked.append(text.replace('.','').replace(',','').strip())
    return masked




# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def get_active_labels(doclist):
    cat_list = set()
    for d in doclist:
        cat_list.update(d.categories)
    return list(cat_list)

def filter_by_categories(doclist, keep_categories):
    catset = frozenset(keep_categories)
    for d in doclist:
        d.categories = list(set(d.categories).intersection(catset))

def __years_to_str(years):
    if isinstance(years, list):
        if len(years) > 1:
            return str(years[0])+'-'+str(years[-1])
        return str(years[0])
    return str(years)


# ----------------------------------------------------------------------------------------------------------------------
# Matrix builders
# ----------------------------------------------------------------------------------------------------------------------
def build_independent_matrices(dataset_name, langs, training_docs, test_docs, label_names, wiki_docs=[], preprocess=True):
    """
    Builds the document-by-term weighted matrices for each language. Representations are independent of each other,
    i.e., each language-specific matrix lies in a dedicate feature space.
    :param dataset_name: the name of the dataset (str)
    :param langs: list of languages (str)
    :param training_docs: map {lang:doc-list} where each doc is a tuple (text, categories, id)
    :param test_docs: map {lang:doc-list} where each doc is a tuple (text, categories, id)
    :param label_names: list of names of labels (str)
    :param wiki_docs: doc-list (optional), if specified, project all wiki docs in the feature spaces built for the languages
    :param preprocess: whether or not to apply language-specific text preprocessing (stopword removal and stemming)
    :return: a MultilingualDataset. If wiki_docs has been specified, a dictionary lW is also returned, which indexes
    by language the processed wikipedia documents in their respective language-specific feature spaces
    """

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    lW = {}

    multilingual_dataset = MultilingualDataset()
    multilingual_dataset.dataset_name = dataset_name
    multilingual_dataset.set_labels(mlb.classes_)
    for lang in langs:
        print("\nprocessing %d training, %d test, %d wiki for language <%s>" %
              (len(training_docs[lang]), len(test_docs[lang]), len(wiki_docs[lang]) if wiki_docs else 0, lang))

        tr_data, tr_labels, IDtr = zip(*training_docs[lang])
        te_data, te_labels, IDte = zip(*test_docs[lang])

        if preprocess:
            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                    tokenizer=NLTKStemTokenizer(lang, verbose=True),
                                    stop_words=stopwords.words(NLTK_LANGMAP[lang]))
        else:
            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)

        Xtr = tfidf.fit_transform(tr_data)
        Xte = tfidf.transform(te_data)
        if wiki_docs:
            lW[lang] = tfidf.transform(wiki_docs[lang])

        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)

        multilingual_dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    multilingual_dataset.show_dimensions()
    multilingual_dataset.show_category_prevalences()

    if wiki_docs:
        return multilingual_dataset, lW
    else:
        return multilingual_dataset


# creates a MultilingualDataset where matrices shares a single yuxtaposed feature space
def build_juxtaposed_matrices(dataset_name, langs, training_docs, test_docs, label_names, preprocess=True):
    """
    Builds the document-by-term weighted matrices for each language. Representations are not independent of each other,
    since all of them lie on the same yuxtaposed feature space.
    :param dataset_name: the name of the dataset (str)
    :param langs: list of languages (str)
    :param training_docs: map {lang:doc-list} where each doc is a tuple (text, categories, id)
    :param test_docs: map {lang:doc-list} where each doc is a tuple (text, categories, id)
    :param label_names: list of names of labels (str)
    :param preprocess: whether or not to apply language-specific text preprocessing (stopword removal and stemming)
    :return: a MultilingualDataset. If wiki_docs has been specified, a dictionary lW is also returned, which indexes
    by language the processed wikipedia documents in their respective language-specific feature spaces
    """

    multiling_dataset = MultilingualDataset()
    multiling_dataset.dataset_name = dataset_name

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    multiling_dataset.set_labels(mlb.classes_)

    tr_data_stack = []
    for lang in langs:
        print("\nprocessing %d training and %d test for language <%s>" % (len(training_docs[lang]), len(test_docs[lang]), lang))
        tr_data, tr_labels, tr_ID = zip(*training_docs[lang])
        te_data, te_labels, te_ID = zip(*test_docs[lang])
        if preprocess:
            tr_data = preprocess_documents(tr_data, lang)
            te_data = preprocess_documents(te_data, lang)
        tr_data_stack.extend(tr_data)
        multiling_dataset.add(lang, tr_data, tr_labels, te_data, te_labels, tr_ID, te_ID)

    tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)
    tfidf.fit(tr_data_stack)

    for lang in langs:
        print("\nweighting documents for language <%s>" % (lang))
        (tr_data, tr_labels, tr_ID), (te_data, te_labels, te_ID) = multiling_dataset[lang]
        Xtr = tfidf.transform(tr_data)
        Xte = tfidf.transform(te_data)
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)
        multiling_dataset.add(lang,Xtr,Ytr,Xte,Yte,tr_ID,te_ID)

    multiling_dataset.show_dimensions()
    return multiling_dataset


# ----------------------------------------------------------------------------------------------------------------------
# Methods to recover the original documents from the MultilingualDataset's ids
# ----------------------------------------------------------------------------------------------------------------------
"""
This method has been added a posteriori, to create document embeddings using the polylingual embeddings of the recent
article 'Word Translation without Parallel Data'; basically, it takes one of the splits and retrieves the RCV documents
from the doc ids and then pickles an object (tr_docs, te_docs, label_names) in the outpath
"""
def retrieve_rcv_documents_from_dataset(datasetpath, rcv1_data_home, rcv2_data_home, outpath):

    tr_ids, te_ids = MultilingualDataset.load_ids(datasetpath)
    assert tr_ids.keys() == te_ids.keys(), 'inconsistent keys tr vs te'
    langs = list(tr_ids.keys())

    print('fetching the datasets')
    rcv1_documents, labels_rcv1 = fetch_RCV1(rcv1_data_home, split='train')
    rcv2_documents, labels_rcv2 = fetch_RCV2(rcv2_data_home, [l for l in langs if l != 'en'])

    filter_by_categories(rcv1_documents, labels_rcv2)
    filter_by_categories(rcv2_documents, labels_rcv1)

    label_names = get_active_labels(rcv1_documents + rcv2_documents)
    print('Active labels in RCV1/2 {}'.format(len(label_names)))

    print('rcv1: {} train, {} test, {} categories'.format(len(rcv1_documents), 0, len(label_names)))
    print('rcv2: {} documents'.format(len(rcv2_documents)), Counter([doc.lang for doc in rcv2_documents]))

    all_docs = rcv1_documents + rcv2_documents
    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    dataset = MultilingualDataset()
    for lang in langs:
        analyzer = CountVectorizer(strip_accents='unicode', min_df=3,
                                   stop_words=stopwords.words(NLTK_LANGMAP[lang])).build_analyzer()

        Xtr,Ytr,IDtr = zip(*[(d.text,d.categories,d.id) for d in all_docs if d.lang == lang and d.id in tr_ids[lang]])
        Xte,Yte,IDte = zip(*[(d.text,d.categories,d.id) for d in all_docs if d.lang == lang and d.id in te_ids[lang]])
        Xtr = [' '.join(analyzer(d)) for d in Xtr]
        Xte = [' '.join(analyzer(d)) for d in Xte]
        Ytr = mlb.transform(Ytr)
        Yte = mlb.transform(Yte)
        dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    dataset.save(outpath)

"""
Same thing but for JRC-Acquis
"""
def retrieve_jrc_documents_from_dataset(datasetpath, jrc_data_home, train_years, test_years, cat_policy, most_common_cat, outpath):

    tr_ids, te_ids = MultilingualDataset.load_ids(datasetpath)
    assert tr_ids.keys() == te_ids.keys(), 'inconsistent keys tr vs te'
    langs = list(tr_ids.keys())

    print('fetching the datasets')

    cat_list = inspect_eurovoc(jrc_data_home, select=cat_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=train_years,
                                                 cat_filter=cat_list, cat_threshold=1, parallel=None,
                                                 most_frequent=most_common_cat)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=test_years, cat_filter=label_names,
                                   parallel='force')

    def filter_by_id(doclist, ids):
        ids_set = frozenset(itertools.chain.from_iterable(ids.values()))
        return [x for x in doclist if (x.parallel_id+'__'+x.id) in ids_set]

    training_docs = filter_by_id(training_docs, tr_ids)
    test_docs = filter_by_id(test_docs, te_ids)

    print('jrc: {} train, {} test, {} categories'.format(len(training_docs), len(test_docs), len(label_names)))

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    dataset = MultilingualDataset()
    for lang in langs:
        analyzer = CountVectorizer(strip_accents='unicode', min_df=3,
                                   stop_words=stopwords.words(NLTK_LANGMAP[lang])).build_analyzer()

        Xtr,Ytr,IDtr = zip(*[(d.text,d.categories,d.parallel_id+'__'+d.id) for d in training_docs if d.lang == lang])
        Xte,Yte,IDte = zip(*[(d.text,d.categories,d.parallel_id+'__'+d.id) for d in test_docs if d.lang == lang])
        Xtr = [' '.join(analyzer(d)) for d in Xtr]
        Xte = [' '.join(analyzer(d)) for d in Xte]
        Ytr = mlb.transform(Ytr)
        Yte = mlb.transform(Yte)
        dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    dataset.save(outpath)

# ----------------------------------------------------------------------------------------------------------------------
# Dataset Generators
# ----------------------------------------------------------------------------------------------------------------------
def prepare_jrc_datasets(jrc_data_home, wiki_data_home, langs, train_years, test_years, cat_policy, most_common_cat=-1, max_wiki=5000, run=0):
    from data.reader.wikipedia_tools import fetch_wikipedia_multilingual, random_wiki_sample


    """
    Prepare all datasets for JRC-Acquis. The datasets include the "feature-independent" version, the
    "feature-yuxtaposed" version, the monolingual version for the UpperBound, and the processed wikipedia matrices.
    In all cases, training documents are strictly non-parallel, and test documents are strictly parallel
    :param jrc_data_home: path to the raw JRC-Acquis documents (it will be downloaded if not found), and the path where
    all splits will be generated
    :param wiki_data_home: path to the wikipedia dump (see data/readers/wikipedia_tools.py)
    :param langs: the list of languages to consider (as defined in data/languages.py)
    :param train_years: a list of ints containing the years to be considered as training documents
    :param test_years: a list of ints containing the  years to be considered as test documents
    :param cat_policy: a string indicating which category selection policy to apply. Valid policies are, e.g., "all"
    (select all categories), "broadest" (select only the broadest concepts in the taxonomy), or "leaves" (select the
    leaves concepts in the taxonomy). See inspect_eurovoc from data/reader/jrcacquis_reader.py for more details
    :param most_common_cat: the maximum number of most common categories to consider, or -1 to keep them all
    :param max_wiki: the maximum number of wikipedia documents to consider (default 5000)
    :param run: a numeric label naming the random split (useful to keep track of different runs)
    :return: None
    """

    name = 'JRCacquis'
    run = '_run' + str(run)
    config_name = 'jrc_nltk_' + __years_to_str(train_years) + \
                  'vs' + __years_to_str(test_years) + \
                  '_' + cat_policy + \
                  ('_top' + str(most_common_cat) if most_common_cat!=-1 else '') + \
                  '_noparallel_processed'

    indep_path = join(jrc_data_home, config_name + run + '.pickle')
    upper_path = join(jrc_data_home, config_name + run + '_upper.pickle')
    yuxta_path = join(jrc_data_home, config_name + run + '_yuxtaposed.pickle')
    wiki_path  = join(jrc_data_home, config_name + run + '.wiki.pickle')
    wiki_docs_path = join(jrc_data_home, config_name + '.wiki.raw.pickle')

    cat_list = inspect_eurovoc(jrc_data_home, select=cat_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=train_years,
                                                 cat_filter=cat_list, cat_threshold=1, parallel=None,
                                                 most_frequent=most_common_cat)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=test_years, cat_filter=label_names,
                                   parallel='force')

    print('Generating feature-independent dataset...')
    training_docs_no_parallel = random_sampling_avoiding_parallel(training_docs)

    def _group_by_lang(doc_list, langs):
        return {lang: [(d.text, d.categories, d.parallel_id + '__' + d.id) for d in doc_list if d.lang == lang]
                for lang in langs}

    training_docs = _group_by_lang(training_docs, langs)
    training_docs_no_parallel = _group_by_lang(training_docs_no_parallel, langs)
    test_docs = _group_by_lang(test_docs, langs)
    if not exists(indep_path):
        wiki_docs=None
        if max_wiki>0:
            if not exists(wiki_docs_path):
                wiki_docs = fetch_wikipedia_multilingual(wiki_data_home, langs, min_words=50, deletions=False)
                wiki_docs = random_wiki_sample(wiki_docs, max_wiki)
                pickle.dump(wiki_docs, open(wiki_docs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            else:
                wiki_docs = pickle.load(open(wiki_docs_path, 'rb'))
            wiki_docs = random_wiki_sample(wiki_docs, max_wiki)

        if wiki_docs:
            lang_data, wiki_docs = build_independent_matrices(name, langs, training_docs_no_parallel, test_docs, label_names, wiki_docs)
            pickle.dump(wiki_docs, open(wiki_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            lang_data = build_independent_matrices(name, langs, training_docs_no_parallel, test_docs, label_names)

        lang_data.save(indep_path)

    print('Generating upper-bound (English-only) dataset...')
    if not exists(upper_path):
        training_docs_eng_only = {'en':training_docs['en']}
        test_docs_eng_only = {'en':test_docs['en']}
        build_independent_matrices(name, ['en'], training_docs_eng_only, test_docs_eng_only, label_names).save(upper_path)

    print('Generating yuxtaposed dataset...')
    if not exists(yuxta_path):
        build_juxtaposed_matrices(name, langs, training_docs_no_parallel, test_docs, label_names).save(yuxta_path)


def prepare_rcv_datasets(outpath, rcv1_data_home, rcv2_data_home, wiki_data_home, langs,
                         train_for_lang=1000, test_for_lang=1000, max_wiki=5000, preprocess=True, run=0):
    from data.reader.wikipedia_tools import fetch_wikipedia_multilingual, random_wiki_sample
    """
        Prepare all datasets for RCV1/RCV2. The datasets include the "feature-independent" version, the
        "feature-yuxtaposed" version, the monolingual version for the UpperBound, and the processed wikipedia matrices.

        :param outpath: path where all splits will be dumped
        :param rcv1_data_home: path to the RCV1-v2 dataset (English only)
        :param rcv2_data_home: path to the RCV2 dataset (all languages other than English)
        :param wiki_data_home: path to the wikipedia dump (see data/readers/wikipedia_tools.py)
        :param langs: the list of languages to consider (as defined in data/languages.py)
        :param train_for_lang: maximum number of training documents per language
        :param test_for_lang:  maximum number of test documents per language
        :param max_wiki: the maximum number of wikipedia documents to consider (default 5000)
        :param preprocess: whether or not to apply language-specific preprocessing (stopwords removal and stemming)
        :param run: a numeric label naming the random split (useful to keep track of different runs)
        :return: None
        """

    assert 'en' in langs, 'English is not in requested languages, but is needed for some datasets'
    assert len(langs)>1, 'the multilingual dataset cannot be built with only one dataset'
    assert not preprocess or set(langs).issubset(set(RCV2_LANGS_WITH_NLTK_STEMMING+['en'])), \
        "languages not in RCV1-v2/RCV2 scope or not in valid for NLTK's processing"

    name = 'RCV1/2'
    run = '_run' + str(run)
    config_name = 'rcv1-2_nltk_trByLang'+str(train_for_lang)+'_teByLang'+str(test_for_lang)+\
                  ('_processed' if preprocess else '_raw')

    indep_path = join(outpath, config_name + run + '.pickle')
    upper_path = join(outpath, config_name + run +'_upper.pickle')
    yuxta_path = join(outpath, config_name + run +'_yuxtaposed.pickle')
    wiki_path = join(outpath, config_name + run + '.wiki.pickle')
    wiki_docs_path = join(outpath, config_name + '.wiki.raw.pickle')

    print('fetching the datasets')
    rcv1_documents, labels_rcv1 = fetch_RCV1(rcv1_data_home, split='train')
    rcv2_documents, labels_rcv2 = fetch_RCV2(rcv2_data_home, [l for l in langs if l!='en'])
    filter_by_categories(rcv1_documents, labels_rcv2)
    filter_by_categories(rcv2_documents, labels_rcv1)

    label_names = get_active_labels(rcv1_documents+rcv2_documents)
    print('Active labels in RCV1/2 {}'.format(len(label_names)))

    print('rcv1: {} train, {} test, {} categories'.format(len(rcv1_documents), 0, len(label_names)))
    print('rcv2: {} documents'.format(len(rcv2_documents)), Counter([doc.lang for doc in rcv2_documents]))

    lang_docs = {lang: [d for d in rcv1_documents + rcv2_documents if d.lang == lang] for lang in langs}

    # for the upper bound there are no parallel versions, so for the English case, we take as many documents as there
    # would be in the multilingual case -- then we will extract from them only train_for_lang for the other cases
    print('Generating upper-bound (English-only) dataset...')
    train, test = train_test_split(lang_docs['en'], train_size=train_for_lang*len(langs), test_size=test_for_lang, shuffle=True)
    train_lang_doc_map = {'en':[(d.text, d.categories, d.id) for d in train]}
    test_lang_doc_map = {'en':[(d.text, d.categories, d.id) for d in test]}
    build_independent_matrices(name, ['en'], train_lang_doc_map, test_lang_doc_map, label_names).save(upper_path)

    train_lang_doc_map['en'] = train_lang_doc_map['en'][:train_for_lang]
    for lang in langs:
        if lang=='en': continue # already split
        test_take = min(test_for_lang, len(lang_docs[lang])-train_for_lang)
        train, test = train_test_split(lang_docs[lang], train_size=train_for_lang, test_size=test_take, shuffle=True)
        train_lang_doc_map[lang] = [(d.text, d.categories, d.id) for d in train]
        test_lang_doc_map[lang]  = [(d.text, d.categories, d.id) for d in test]

    print('Generating feature-independent dataset...')
    wiki_docs=None
    if max_wiki>0:
        if not exists(wiki_docs_path):
            wiki_docs = fetch_wikipedia_multilingual(wiki_data_home, langs, min_words=50, deletions=False)
            wiki_docs = random_wiki_sample(wiki_docs, max_wiki)
            pickle.dump(wiki_docs, open(wiki_docs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            wiki_docs = pickle.load(open(wiki_docs_path, 'rb'))
        wiki_docs = random_wiki_sample(wiki_docs, max_wiki)

    if wiki_docs:
        lang_data, wiki_docs_matrix = build_independent_matrices(name, langs, train_lang_doc_map, test_lang_doc_map, label_names, wiki_docs, preprocess)
        pickle.dump(wiki_docs_matrix, open(wiki_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        lang_data = build_independent_matrices(name, langs, train_lang_doc_map, test_lang_doc_map, label_names, wiki_docs, preprocess)

    lang_data.save(indep_path)

    print('Generating yuxtaposed dataset...')
    build_juxtaposed_matrices(name, langs, train_lang_doc_map, test_lang_doc_map, label_names, preprocess).save(yuxta_path)


# ----------------------------------------------------------------------------------------------------------------------
# Methods to generate full RCV and JRC datasets
# ----------------------------------------------------------------------------------------------------------------------
def full_rcv_(rcv1_data_home, rcv2_data_home, outpath, langs):


    print('fetching the datasets')
    rcv1_train_documents, labels_rcv1 = fetch_RCV1(rcv1_data_home, split='train')
    rcv1_test_documents, labels_rcv1_test = fetch_RCV1(rcv1_data_home, split='test')
    rcv2_documents, labels_rcv2 = fetch_RCV2(rcv2_data_home, [l for l in langs if l != 'en'])

    filter_by_categories(rcv1_train_documents, labels_rcv2)
    filter_by_categories(rcv1_test_documents, labels_rcv2)
    filter_by_categories(rcv2_documents, labels_rcv1)

    label_names = get_active_labels(rcv1_train_documents + rcv2_documents)
    print('Active labels in RCV1/2 {}'.format(len(label_names)))

    print('rcv1: {} train, {} test, {} categories'.format(len(rcv1_train_documents), len(rcv1_test_documents), len(label_names)))
    print('rcv2: {} documents'.format(len(rcv2_documents)), Counter([doc.lang for doc in rcv2_documents]))

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    all_docs = rcv1_train_documents + rcv1_test_documents + rcv2_documents
    lang_docs = {lang: [d for d in all_docs if d.lang == lang] for lang in langs}

    def get_ids(doclist):
        return frozenset([d.id for d in doclist])

    tr_ids = {'en': get_ids(rcv1_train_documents)}
    te_ids = {'en': get_ids(rcv1_test_documents)}
    for lang in langs:
        if lang == 'en': continue
        tr_ids[lang], te_ids[lang] = train_test_split([d.id for d in lang_docs[lang]], test_size=.3)

    dataset = MultilingualDataset()
    dataset.dataset_name = 'RCV1/2-full'
    for lang in langs:
        print(f'processing {lang} with {len(tr_ids[lang])} training documents and {len(te_ids[lang])} documents')
        analyzer = CountVectorizer(
            strip_accents='unicode', min_df=3, stop_words=stopwords.words(NLTK_LANGMAP[lang])
        ).build_analyzer()

        Xtr,Ytr,IDtr = zip(*[(d.text,d.categories,d.id) for d in lang_docs[lang] if d.id in tr_ids[lang]])
        Xte,Yte,IDte = zip(*[(d.text,d.categories,d.id) for d in lang_docs[lang] if d.id in te_ids[lang]])
        Xtr = [' '.join(analyzer(d)) for d in Xtr]
        Xte = [' '.join(analyzer(d)) for d in Xte]
        Ytr = mlb.transform(Ytr)
        Yte = mlb.transform(Yte)
        dataset.add(lang, _mask_numbers(Xtr), Ytr, _mask_numbers(Xte), Yte, IDtr, IDte)

    dataset.save(outpath)


def full_jrc_(jrc_data_home, langs, train_years, test_years, outpath, cat_policy='all', most_common_cat=300):

    print('fetching the datasets')
    cat_list = inspect_eurovoc(jrc_data_home, select=cat_policy)
    training_docs, label_names = fetch_jrcacquis(
        langs=langs, data_path=jrc_data_home, years=train_years, cat_filter=cat_list, cat_threshold=1, parallel=None, most_frequent=most_common_cat
    )
    test_docs, _ = fetch_jrcacquis(
        langs=langs, data_path=jrc_data_home, years=test_years, cat_filter=label_names, parallel='force'
    )

    def _group_by_lang(doc_list, langs):
        return {lang: [d for d in doc_list if d.lang == lang] for lang in langs}

    training_docs = _group_by_lang(training_docs, langs)
    test_docs = _group_by_lang(test_docs, langs)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    dataset = MultilingualDataset()
    data.dataset_name = 'JRC-Acquis-full'
    for lang in langs:
        analyzer = CountVectorizer(
            strip_accents='unicode', min_df=3, stop_words=stopwords.words(NLTK_LANGMAP[lang])
        ).build_analyzer()

        Xtr, Ytr, IDtr = zip(*[(d.text, d.categories, d.parallel_id + '__' + d.id) for d in training_docs[lang] if d.lang == lang])
        Xte, Yte, IDte = zip(*[(d.text, d.categories, d.parallel_id + '__' + d.id) for d in test_docs[lang] if d.lang == lang])
        Xtr = [' '.join(analyzer(d)) for d in Xtr]
        Xte = [' '.join(analyzer(d)) for d in Xte]
        Ytr = mlb.transform(Ytr)
        Yte = mlb.transform(Yte)
        dataset.add(lang, _mask_numbers(Xtr), Ytr, _mask_numbers(Xte), Yte, IDtr, IDte)

    dataset.save(outpath)


#-----------------------------------------------------------------------------------------------------------------------
# MAIN BUILDER
#-----------------------------------------------------------------------------------------------------------------------

if __name__=='__main__':
    import sys
    RCV1_PATH = '../Datasets/RCV1-v2/unprocessed_corpus'
    RCV2_PATH = '../Datasets/RCV2'
    JRC_DATAPATH = "../Datasets/JRC_Acquis_v3"
    full_rcv_(RCV1_PATH, RCV2_PATH, outpath='../rcv2/rcv1-2_doclist_full_processed.pickle', langs=RCV2_LANGS_WITH_NLTK_STEMMING + ['en'])
    # full_jrc_(JRC_DATAPATH, lang_set['JRC_NLTK'], train_years=list(range(1958, 2006)), test_years=[2006], outpath='../jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle', cat_policy='all', most_common_cat=300)
    sys.exit(0)

    # datasetpath = '../jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle' # '../rcv2/rcv1-2_doclist_full_processed.pickle'
    # data = MultilingualDataset.load(datasetpath)
    # data.dataset_name='JRC-Acquis-full'#'RCV1/2-full'
    # for lang in RCV2_LANGS_WITH_NLTK_STEMMING + ['en']:
    #     (Xtr, ytr, idtr), (Xte, yte, idte) = data.multiling_dataset[lang]
    #     data.multiling_dataset[lang] = ((_mask_numbers(Xtr), ytr, idtr), (_mask_numbers(Xte), yte, idte))
    # data.save('../jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle')#'../rcv2/rcv1-2_doclist_full_processed_2.pickle')
    # sys.exit(0)

    assert len(sys.argv) == 5, "wrong number of arguments; required: " \
                               "<JRC_PATH> <RCV1_PATH> <RCV2_PATH> <WIKI_PATH> "

    JRC_DATAPATH = sys.argv[1] # "../Datasets/JRC_Acquis_v3"
    RCV1_PATH = sys.argv[2] #'../Datasets/RCV1-v2/unprocessed_corpus'
    RCV2_PATH = sys.argv[3] #'../Datasets/RCV2'
    WIKI_DATAPATH = sys.argv[4] #"../Datasets/Wikipedia/multilingual_docs_JRC_NLTK"

    langs = lang_set['JRC_NLTK']
    max_wiki = 5000

    for run in range(0,10):
        print('Building JRC-Acquis datasets run', run)
        prepare_jrc_datasets(JRC_DATAPATH, WIKI_DATAPATH, langs,
                             train_years=list(range(1958, 2006)), test_years=[2006], max_wiki=max_wiki,
                             cat_policy='all', most_common_cat=300, run=run)

        print('Building RCV1-v2/2 datasets run', run)
        prepare_rcv_datasets(RCV2_PATH, RCV1_PATH, RCV2_PATH, WIKI_DATAPATH, RCV2_LANGS_WITH_NLTK_STEMMING + ['en'],
                             train_for_lang=1000, test_for_lang=1000, max_wiki=max_wiki, run=run)

        # uncomment this code if you want to retrieve the original documents to generate the data splits for PLE
        # (make sure you have not modified the above parameters, or adapt the following paths accordingly...)
        # datasetpath = join(RCV2_PATH,'rcv1-2_nltk_trByLang1000_teByLang1000_processed_run{}.pickle'.format(run))
        # outpath = datasetpath.replace('_nltk_','_doclist_')
        # retrieve_rcv_documents_from_dataset(datasetpath, RCV1_PATH, RCV2_PATH, outpath)

        # datasetpath = join(JRC_DATAPATH, 'jrc_nltk_1958-2005vs2006_all_top300_noparallel_processed_run{}.pickle'.format(run))
        # outpath = datasetpath.replace('_nltk_', '_doclist_')
        # retrieve_jrc_documents_from_dataset(datasetpath, JRC_DATAPATH, train_years=list(range(1958, 2006)), test_years=[2006], cat_policy='all', most_common_cat=300, outpath=outpath)



