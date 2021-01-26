import re
import xml.etree.ElementTree as ET
from os.path import join, exists
from zipfile import ZipFile

import numpy as np

from src.util.file import download_file_if_not_exists
from src.util.file import list_files

"""
RCV2's Nomenclature:
ru = Russian
da = Danish
de = German
es = Spanish
lat = Spanish Latin-American (actually is also 'es' in the collection)
fr = French
it = Italian
nl = Dutch
pt = Portuguese
sv = Swedish
ja = Japanese
htw = Chinese
no = Norwegian
"""

RCV1_TOPICHIER_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig"
RCV1PROC_BASE_URL= 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files'
RCV1_BASE_URL = "http://www.daviddlewis.com/resources/testcollections/rcv1/"
RCV2_BASE_URL = "http://trec.nist.gov/data/reuters/reuters.html"

rcv1_test_data_gz = ['lyrl2004_tokens_test_pt0.dat.gz',
             'lyrl2004_tokens_test_pt1.dat.gz',
             'lyrl2004_tokens_test_pt2.dat.gz',
             'lyrl2004_tokens_test_pt3.dat.gz']

rcv1_train_data_gz = ['lyrl2004_tokens_train.dat.gz']

rcv1_doc_cats_data_gz = 'rcv1-v2.topics.qrels.gz'

RCV2_LANG_DIR = {'ru':'REUTE000',
                 'de':'REUTE00A',
                 'fr':'REUTE00B',
                 'sv':'REUTE001',
                 'no':'REUTE002',
                 'da':'REUTE003',
                 'pt':'REUTE004',
                 'it':'REUTE005',
                 'es':'REUTE006',
                 'lat':'REUTE007',
                 'jp':'REUTE008',
                 'htw':'REUTE009',
                 'nl':'REUTERS_'}


class RCV_Document:

    def __init__(self, id, text, categories, date='', lang=None):
        self.id = id
        self.date = date
        self.lang = lang
        self.text = text
        self.categories = categories


class ExpectedLanguageException(Exception): pass
class IDRangeException(Exception): pass


nwords = []

def parse_document(xml_content, assert_lang=None, valid_id_range=None):
    root = ET.fromstring(xml_content)
    if assert_lang:
        if assert_lang not in root.attrib.values():
            if assert_lang != 'jp' or 'ja' not in root.attrib.values():  # some documents are attributed to 'ja', others to 'jp'
                raise ExpectedLanguageException('error: document of a different language')

    doc_id = root.attrib['itemid']
    if valid_id_range is not None:
        if not valid_id_range[0] <= int(doc_id) <= valid_id_range[1]:
            raise IDRangeException

    doc_categories = [cat.attrib['code'] for cat in
                      root.findall('.//metadata/codes[@class="bip:topics:1.0"]/code')]

    doc_date = root.attrib['date']
    doc_title = root.find('.//title').text
    doc_headline = root.find('.//headline').text
    doc_body = '\n'.join([p.text for p in root.findall('.//text/p')])

    if not doc_body:
        raise ValueError('Empty document')

    if doc_title is None: doc_title = ''
    if doc_headline is None or doc_headline in doc_title: doc_headline = ''
    text = '\n'.join([doc_title, doc_headline, doc_body]).strip()

    text_length = len(text.split())
    global nwords
    nwords.append(text_length)

    return RCV_Document(id=doc_id, text=text, categories=doc_categories, date=doc_date, lang=assert_lang)


def fetch_RCV1(data_path, split='all'):

    assert split in ['train', 'test', 'all'], 'split should be "train", "test", or "all"'

    request = []
    labels = set()
    read_documents = 0
    lang = 'en'

    training_documents = 23149
    test_documents = 781265

    if split == 'all':
        split_range = (2286, 810596)
        expected = training_documents+test_documents
    elif split == 'train':
        split_range = (2286, 26150)
        expected = training_documents
    else:
        split_range = (26151, 810596)
        expected = test_documents

    global nwords
    nwords=[]
    for part in list_files(data_path):
        if not re.match('\d+\.zip', part): continue
        target_file = join(data_path, part)
        assert exists(target_file), \
            "You don't seem to have the file "+part+" in " + data_path + ", and the RCV1 corpus can not be downloaded"+\
            " w/o a formal permission. Please, refer to " + RCV1_BASE_URL + " for more information."
        zipfile = ZipFile(target_file)
        for xmlfile in zipfile.namelist():
            xmlcontent = zipfile.open(xmlfile).read()
            try:
                doc = parse_document(xmlcontent, assert_lang=lang, valid_id_range=split_range)
                labels.update(doc.categories)
                request.append(doc)
                read_documents += 1
            except ValueError:
                print('\n\tskipping document {} with inconsistent language label: expected language {}'.format(part+'/'+xmlfile, lang))
            except (IDRangeException, ExpectedLanguageException) as e:
                pass
            print('\r[{}] read {} documents'.format(part, len(request)), end='')
            if read_documents == expected: break
        if read_documents == expected: break
    print()
    print('ave:{} std {} min {} max {}'.format(np.mean(nwords), np.std(nwords), np.min(nwords), np.max(nwords)))
    return request, list(labels)


def fetch_RCV2(data_path, languages=None):

    if not languages:
        languages = list(RCV2_LANG_DIR.keys())
    else:
        assert set(languages).issubset(set(RCV2_LANG_DIR.keys())), 'languages not in scope'

    request = []
    labels = set()
    global nwords
    nwords=[]
    for lang in languages:
        path = join(data_path, RCV2_LANG_DIR[lang])
        lang_docs_read = 0
        for part in list_files(path):
            target_file = join(path, part)
            assert exists(target_file), \
                "You don't seem to have the file "+part+" in " + path + ", and the RCV2 corpus can not be downloaded"+\
                " w/o a formal permission. Please, refer to " + RCV2_BASE_URL + " for more information."
            zipfile = ZipFile(target_file)
            for xmlfile in zipfile.namelist():
                xmlcontent = zipfile.open(xmlfile).read()
                try:
                    doc = parse_document(xmlcontent, assert_lang=lang)
                    labels.update(doc.categories)
                    request.append(doc)
                    lang_docs_read += 1
                except ValueError:
                    print('\n\tskipping document {} with inconsistent language label: expected language {}'.format(RCV2_LANG_DIR[lang]+'/'+part+'/'+xmlfile, lang))
                except (IDRangeException, ExpectedLanguageException) as e:
                    pass
                print('\r[{}] read {} documents, {} for language {}'.format(RCV2_LANG_DIR[lang]+'/'+part, len(request), lang_docs_read, lang), end='')
        print()
    print('ave:{} std {} min {} max {}'.format(np.mean(nwords), np.std(nwords), np.min(nwords), np.max(nwords)))
    return request, list(labels)


def fetch_topic_hierarchy(path, topics='all'):
    assert topics in ['all', 'leaves']

    download_file_if_not_exists(RCV1_TOPICHIER_URL, path)
    hierarchy = {}
    for line in open(path, 'rt'):
        parts = line.strip().split()
        parent,child = parts[1],parts[3]
        if parent not in hierarchy:
            hierarchy[parent]=[]
        hierarchy[parent].append(child)

    del hierarchy['None']
    del hierarchy['Root']
    print(hierarchy)

    if topics=='all':
        topics = set(hierarchy.keys())
        for parent in hierarchy.keys():
            topics.update(hierarchy[parent])
        return list(topics)
    elif topics=='leaves':
        parents = set(hierarchy.keys())
        childs = set()
        for parent in hierarchy.keys():
            childs.update(hierarchy[parent])
        return list(childs.difference(parents))


