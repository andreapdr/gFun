from __future__ import print_function

import os
import pickle
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from os.path import join
from random import shuffle

import rdflib
from rdflib.namespace import RDF, SKOS
from sklearn.datasets import get_data_home

from src.data.languages import JRC_LANGS
from src.data.languages import lang_set
from src.util.file import download_file, list_dirs, list_files

"""
JRC Acquis' Nomenclature:
bg = Bulgarian
cs = Czech
da = Danish
de = German
el = Greek
en = English
es = Spanish
et = Estonian
fi = Finnish
fr = French
hu = Hungarian
it = Italian
lt = Lithuanian
lv = Latvian
nl = Dutch
mt = Maltese
pl = Polish
pt = Portuguese
ro = Romanian
sk = Slovak
sl = Slovene
sv = Swedish
"""

class JRCAcquis_Document:
    def __init__(self, id, name, lang, year, head, body, categories):
        self.id = id
        self.parallel_id = name
        self.lang = lang
        self.year = year
        self.text = body if not head else head + "\n" + body
        self.categories = categories

# this is a workaround... for some reason, acutes are codified in a non-standard manner in titles
# however, it seems that the title is often appearing as the first paragraph in the text/body (with
# standard codification), so it might be preferable not to read the header after all (as here by default)
def _proc_acute(text):
    for ch in ['a','e','i','o','u']:
        text = text.replace('%'+ch+'acute%',ch)
    return text

def parse_document(file, year, head=False):
    root = ET.parse(file).getroot()

    doc_name = root.attrib['n'] # e.g., '22006A0211(01)'
    doc_lang = root.attrib['lang'] # e.g., 'es'
    doc_id   = root.attrib['id'] # e.g., 'jrc22006A0211_01-es'
    doc_categories = [cat.text for cat in root.findall('.//teiHeader/profileDesc/textClass/classCode[@scheme="eurovoc"]')]
    doc_head = _proc_acute(root.find('.//text/body/head').text) if head else ''
    doc_body = '\n'.join([p.text for p in root.findall('.//text/body/div[@type="body"]/p')])

    def raise_if_empty(field, from_file):
        if isinstance(field, str):
            if not field.strip():
                raise ValueError("Empty field in file %s" % from_file)

    raise_if_empty(doc_name, file)
    raise_if_empty(doc_lang, file)
    raise_if_empty(doc_id, file)
    if head: raise_if_empty(doc_head, file)
    raise_if_empty(doc_body, file)

    return JRCAcquis_Document(id=doc_id, name=doc_name, lang=doc_lang, year=year, head=doc_head, body=doc_body, categories=doc_categories)

# removes documents without a counterpart in all other languages
def _force_parallel(doclist, langs):
    n_langs = len(langs)
    par_id_count = Counter([d.parallel_id for d in doclist])
    parallel_doc_ids = set([id for id,count in par_id_count.items() if count==n_langs])
    return [doc for doc in doclist if doc.parallel_id in parallel_doc_ids]

def random_sampling_avoiding_parallel(doclist):
    random_order = list(range(len(doclist)))
    shuffle(random_order)
    sampled_request = []
    parallel_ids = set()
    for ind in random_order:
        pid = doclist[ind].parallel_id
        if pid not in parallel_ids:
            sampled_request.append(doclist[ind])
            parallel_ids.add(pid)
    print('random_sampling_no_parallel:: from {} documents to {} documents'.format(len(doclist), len(sampled_request)))
    return sampled_request


#filters out documents which do not contain any category in the cat_filter list, and filter all labels not in cat_filter
def _filter_by_category(doclist, cat_filter):
    if not isinstance(cat_filter, frozenset):
        cat_filter = frozenset(cat_filter)
    filtered = []
    for doc in doclist:
        doc.categories = list(cat_filter & set(doc.categories))
        if doc.categories:
            doc.categories.sort()
            filtered.append(doc)
    print("filtered %d documents out without categories in the filter list" % (len(doclist) - len(filtered)))
    return filtered

#filters out categories with less than cat_threshold documents (and filters documents containing those categories)
def _filter_by_frequency(doclist, cat_threshold):
    cat_count = Counter()
    for d in doclist:
        cat_count.update(d.categories)

    freq_categories = [cat for cat,count in cat_count.items() if count>cat_threshold]
    freq_categories.sort()
    return _filter_by_category(doclist, freq_categories), freq_categories

#select top most_frequent categories (and filters documents containing those categories)
def _most_common(doclist, most_frequent):
    cat_count = Counter()
    for d in doclist:
        cat_count.update(d.categories)

    freq_categories = [cat for cat,count in cat_count.most_common(most_frequent)]
    freq_categories.sort()
    return _filter_by_category(doclist, freq_categories), freq_categories

def _get_categories(request):
    final_cats = set()
    for d in request:
        final_cats.update(d.categories)
    return list(final_cats)

def fetch_jrcacquis(langs=None, data_path=None, years=None, ignore_unclassified=True, cat_filter=None, cat_threshold=0,
                    parallel=None, most_frequent=-1, DOWNLOAD_URL_BASE ='http://optima.jrc.it/Acquis/JRC-Acquis.3.0/corpus/'):

    assert parallel in [None, 'force', 'avoid'], 'parallel mode not supported'
    if not langs:
        langs = JRC_LANGS
    else:
        if isinstance(langs, str): langs = [langs]
        for l in langs:
            if l not in JRC_LANGS:
                raise ValueError('Language %s is not among the valid languages in JRC-Acquis v3' % l)

    if not data_path:
        data_path = get_data_home()

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    request = []
    total_read = 0
    for l in langs:
        file_name = 'jrc-'+l+'.tgz'
        archive_path = join(data_path, file_name)

        if not os.path.exists(archive_path):
            print("downloading language-specific dataset (once and for all) into %s" % data_path)
            DOWNLOAD_URL = join(DOWNLOAD_URL_BASE, file_name)
            download_file(DOWNLOAD_URL, archive_path)
            print("untarring dataset...")
            tarfile.open(archive_path, 'r:gz').extractall(data_path)

        documents_dir = join(data_path, l)

        print("Reading documents...")
        read = 0
        for dir in list_dirs(documents_dir):
            year = int(dir)
            if years==None or year in years:
                year_dir = join(documents_dir,dir)
                pickle_name = join(data_path, 'jrc_' + l + '_' + dir + '.pickle')
                if os.path.exists(pickle_name):
                    print("loading from file %s" % pickle_name)
                    l_y_documents = pickle.load(open(pickle_name, "rb"))
                    read += len(l_y_documents)
                else:
                    l_y_documents = []
                    all_documents = list_files(year_dir)
                    empty = 0
                    for i,doc_file in enumerate(all_documents):
                        try:
                            jrc_doc = parse_document(join(year_dir, doc_file), year)
                        except ValueError:
                            jrc_doc = None

                        if jrc_doc and (not ignore_unclassified or jrc_doc.categories):
                            l_y_documents.append(jrc_doc)
                        else: empty += 1
                        if len(all_documents)>50 and ((i+1) % (len(all_documents)/50) == 0):
                            print('\r\tfrom %s: completed %d%%' % (year_dir, int((i+1)*100.0/len(all_documents))), end='')
                        read+=1
                    print('\r\tfrom %s: completed 100%% read %d documents (discarded %d without categories or empty fields)\n' % (year_dir, i+1, empty), end='')
                    print("\t\t(Pickling object for future runs in %s)" % pickle_name)
                    pickle.dump(l_y_documents, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)
                request += l_y_documents
        print("Read %d documents for language %s\n" % (read, l))
        total_read += read
    print("Read %d documents in total" % (total_read))

    if parallel=='force':
        request = _force_parallel(request, langs)
    elif parallel == 'avoid':
        request = random_sampling_avoiding_parallel(request)

    final_cats = _get_categories(request)

    if cat_filter:
        request = _filter_by_category(request, cat_filter)
        final_cats = _get_categories(request)
    if cat_threshold > 0:
        request, final_cats = _filter_by_frequency(request, cat_threshold)
    if most_frequent != -1 and len(final_cats) > most_frequent:
        request, final_cats = _most_common(request, most_frequent)

    return request, final_cats

def print_cat_analysis(request):
    cat_count = Counter()
    for d in request:
        cat_count.update(d.categories)
    print("Number of active categories: {}".format(len(cat_count)))
    print(cat_count.most_common())

# inspects the Eurovoc thesaurus in order to select a subset of categories
# currently, only 'broadest' policy (i.e., take all categories with no parent category), and 'all' is implemented
def inspect_eurovoc(data_path, eurovoc_skos_core_concepts_filename='eurovoc_in_skos_core_concepts.rdf',
                    eurovoc_url="http://publications.europa.eu/mdr/resource/thesaurus/eurovoc-20160630-0/skos/eurovoc_in_skos_core_concepts.zip",
                    select="broadest"):

    fullpath_pickle = join(data_path, select+'_concepts.pickle')
    if os.path.exists(fullpath_pickle):
        print("Pickled object found in %s. Loading it." % fullpath_pickle)
        return pickle.load(open(fullpath_pickle,'rb'))

    fullpath = join(data_path, eurovoc_skos_core_concepts_filename)
    if not os.path.exists(fullpath):
        print("Path %s does not exist. Trying to download the skos EuroVoc file from %s" % (data_path, eurovoc_url))
        download_file(eurovoc_url, fullpath)
        print("Unzipping file...")
        zipped = zipfile.ZipFile(data_path + '.zip', 'r')
        zipped.extract("eurovoc_in_skos_core_concepts.rdf", data_path)
        zipped.close()

    print("Parsing %s" %fullpath)
    g = rdflib.Graph()
    g.parse(location=fullpath, format="application/rdf+xml")

    if select == "all":
        print("Selecting all concepts")
        all_concepts = list(g.subjects(RDF.type, SKOS.Concept))
        all_concepts = [c.toPython().split('/')[-1] for c in all_concepts]
        all_concepts.sort()
        selected_concepts = all_concepts
    elif select=="broadest":
        print("Selecting broadest concepts (those without any other broader concept linked to it)")
        all_concepts = set(g.subjects(RDF.type, SKOS.Concept))
        narrower_concepts = set(g.subjects(SKOS.broader, None))
        broadest_concepts = [c.toPython().split('/')[-1] for c in (all_concepts - narrower_concepts)]
        broadest_concepts.sort()
        selected_concepts = broadest_concepts
    elif select=="leaves":
        print("Selecting leaves concepts (those not linked as broader of any other concept)")
        all_concepts = set(g.subjects(RDF.type, SKOS.Concept))
        broad_concepts = set(g.objects(None, SKOS.broader))
        leave_concepts = [c.toPython().split('/')[-1] for c in (all_concepts - broad_concepts)]
        leave_concepts.sort()
        selected_concepts = leave_concepts
    else:
        raise ValueError("Selection policy %s is not currently supported" % select)

    print("%d %s concepts found" % (len(selected_concepts), leave_concepts))
    print("Pickling concept list for faster further requests in %s" % fullpath_pickle)
    pickle.dump(selected_concepts, open(fullpath_pickle, 'wb'), pickle.HIGHEST_PROTOCOL)

    return selected_concepts

if __name__ == '__main__':

    def single_label_fragment(doclist):
        single = [d for d in doclist if len(d.categories) < 2]
        final_categories = set([d.categories[0] if d.categories else [] for d in single])
        print('{} single-label documents ({} categories) from the original {} documents'.format(len(single),
                                                                                                len(final_categories),
                                                                                                len(doclist)))
        return single, list(final_categories)

    train_years = list(range(1986, 2006))
    test_years = [2006]
    cat_policy = 'leaves'
    most_common_cat = 300
    # JRC_DATAPATH = "/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3"
    JRC_DATAPATH = "/storage/andrea/FUNNELING/data/JRC_Acquis_v3"
    langs = lang_set['JRC_NLTK']
    cat_list = inspect_eurovoc(JRC_DATAPATH, select=cat_policy)
    sys.exit()

    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=JRC_DATAPATH, years=train_years,cat_filter=cat_list, cat_threshold=1, parallel=None,most_frequent=most_common_cat)
    test_docs, label_namestest = fetch_jrcacquis(langs=langs, data_path=JRC_DATAPATH, years=test_years, cat_filter=label_names,parallel='force')

    print('JRC-train: {} documents, {} labels'.format(len(training_docs), len(label_names)))
    print('JRC-test: {} documents, {} labels'.format(len(test_docs), len(label_namestest)))

    training_docs, label_names = single_label_fragment(training_docs)
    test_docs, label_namestest = single_label_fragment(test_docs)

    print('JRC-train: {} documents, {} labels'.format(len(training_docs), len(label_names)))
    print('JRC-test: {} documents, {} labels'.format(len(test_docs), len(label_namestest)))


