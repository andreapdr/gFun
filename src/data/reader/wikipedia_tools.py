from __future__ import print_function

# import ijson
# from ijson.common import ObjectBuilder
import os
import pickle
import re
from bz2 import BZ2File
from itertools import islice
from os.path import join
from xml.sax.saxutils import escape

import numpy as np
from util.file import list_dirs, list_files

policies = ["IN_ALL_LANGS", "IN_ANY_LANG"]

"""
This file contains a set of tools for processing the Wikipedia multilingual documents.
In what follows, it is assumed that you have already downloaded a Wikipedia dump (https://dumps.wikimedia.org/)
and have processed each document to clean their texts with one of the tools:
    - https://github.com/aesuli/wikipediatools (Python 2)
    - https://github.com/aesuli/wikipedia-extractor (Python 3)
It is also assumed you have dowloaded the all-entities json file (e.g., https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2)

This tools help you in:
    - Processes the huge json file as a stream, and create a multilingual map of corresponding titles for each language.
    Set the policy = "IN_ALL_LANGS" will extract only titles which appear in all (AND) languages, whereas "IN_ANY_LANG"
    extracts all titles appearing in at least one (OR) language (warning: this will creates a huge dictionary).
    Note: This version is quite slow. Although it is run once for all, you might be prefer to take a look at "Wikidata in BigQuery".
    - Processes the huge json file as a stream a creates a simplified file which occupies much less and is far faster to be processed.
    - Use the multilingual map to extract, from the clean text versions, individual xml documents containing all
    language-specific versions from the document.
    - Fetch the multilingual documents to create, for each of the specified languages, a list containing all documents,
    in a way that the i-th element from any list refers to the same element in the respective language.
"""

def _doc_generator(text_path, langs):
    dotspace = re.compile(r'\.(?!\s)')
    for l,lang in enumerate(langs):
        print("Processing language <%s> (%d/%d)" % (lang, l, len(langs)))
        lang_dir = join(text_path, lang)
        split_dirs = list_dirs(lang_dir)
        for sd,split_dir in enumerate(split_dirs):
            print("\tprocessing split_dir <%s> (%d/%d)" % (split_dir, sd, len(split_dirs)))
            split_files = list_files(join(lang_dir, split_dir))
            for sf,split_file in enumerate(split_files):
                print("\t\tprocessing split_file <%s> (%d/%d)" % (split_file, sf, len(split_files)))
                with BZ2File(join(lang_dir, split_dir, split_file), 'r', buffering=1024*1024) as fi:
                    while True:
                        doc_lines = list(islice(fi, 3))
                        if doc_lines:
                            # some sentences are not followed by a space after the dot
                            doc_lines[1] = dotspace.sub('. ', doc_lines[1])
                            # [workaround] I found &nbsp; html symbol was not treated, and unescaping it now might not help...
                            doc_lines[1] = escape(doc_lines[1].replace("&nbsp;", " "))
                            yield doc_lines, lang
                        else: break

def _extract_title(doc_lines):
    m = re.search('title="(.+?)"', doc_lines[0])
    if m: return m.group(1).decode('utf-8')
    else: raise ValueError("Error in xml format: document head is %s" % doc_lines[0])

def _create_doc(target_file, id, doc, lang):
    doc[0] = doc[0][:-2] + (' lang="%s">\n'%lang)
    with open(target_file, 'w') as fo:
        fo.write('<multidoc id="%s">\n'%id)
        [fo.write(line) for line in doc]
        fo.write('</multidoc>')

def _append_doc(target_file, doc, lang):
    doc[0] = doc[0][:-2] + (' lang="%s">\n' % lang)
    with open(target_file, 'r', buffering=1024*1024) as fi:
        lines = fi.readlines()
    if doc[0] in lines[1::3]:
        return
    lines[-1:-1]=doc
    with open(target_file, 'w', buffering=1024*1024) as fo:
        [fo.write(line) for line in lines]

def extract_multilingual_documents(inv_dict, langs, text_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for lang in langs:
        if lang not in inv_dict:
            raise ValueError("Lang %s is not in the dictionary" % lang)

    docs_created = len(list_files(out_path))
    print("%d multilingual documents found." % docs_created)
    for doc,lang in _doc_generator(text_path, langs):
        title = _extract_title(doc)

        if title in inv_dict[lang]:
            #pass
            ids = inv_dict[lang][title]
            for id in ids:
                target_file = join(out_path, id) + ".xml"
                if os.path.exists(target_file):
                    _append_doc(target_file, doc, lang)
                else:
                    _create_doc(target_file, id, doc, lang)
                    docs_created+=1
        else:
            if not re.match('[A-Za-z]+', title):
                print("Title <%s> for lang <%s> not in dictionary" % (title, lang))



def extract_multilingual_titles_from_simplefile(data_dir, filename, langs, policy="IN_ALL_LANGS", return_both=True):
    simplified_file = join(data_dir,filename)

    if policy not in policies:
        raise ValueError("Policy %s not supported." % policy)
    print("extracting multilingual titles with policy %s (%s)" % (policy,' '.join(langs)))

    lang_prefix = list(langs)
    lang_prefix.sort()
    pickle_prefix = "extraction_" + "_".join(lang_prefix) + "." + policy
    pickle_dict = join(data_dir, pickle_prefix+".multi_dict.pickle")
    pickle_invdict = join(data_dir, pickle_prefix+".multi_invdict.pickle")
    if os.path.exists(pickle_invdict):
        if return_both and os.path.exists(pickle_dict):
            print("Pickled files found in %s. Loading both (direct and inverse dictionaries)." % data_dir)
            return pickle.load(open(pickle_dict, 'rb')), pickle.load(open(pickle_invdict, 'rb'))
        elif return_both==False:
            print("Pickled file found in %s. Loading inverse dictionary only." % pickle_invdict)
            return pickle.load(open(pickle_invdict, 'rb'))

    multiling_titles = {}
    inv_dict = {lang:{} for lang in langs}

    def process_entry(line):
        parts = line.strip().split('\t')
        id = parts[0]
        if id in multiling_titles:
            raise ValueError("id <%s> already indexed" % id)

        titles = dict(((lang_title[:lang_title.find(':')],lang_title[lang_title.find(':')+1:].decode('utf-8')) for lang_title in parts[1:]))
        for lang in titles.keys():
            if lang not in langs:
                del titles[lang]

        if (policy == "IN_ALL_LANGS" and len(titles) == len(langs))\
                or (policy == "IN_ANY_LANG" and len(titles) > 0):
            multiling_titles[id] = titles
            for lang, title in titles.items():
                if title in inv_dict[lang]:
                    inv_dict[lang][title].append(id)
                inv_dict[lang][title] = [id]

    with BZ2File(simplified_file, 'r', buffering=1024*1024*16) as fi:
        completed = 0
        try:
            for line in fi:
                process_entry(line)
                completed += 1
                if completed % 10 == 0:
                    print("\rCompleted %d\ttitles %d" % (completed,len(multiling_titles)), end="")
            print("\rCompleted %d\t\ttitles %d" % (completed, len(multiling_titles)), end="\n")
        except EOFError:
            print("\nUnexpected file ending... saving anyway")

        print("Pickling dictionaries in %s" % data_dir)
        pickle.dump(multiling_titles, open(pickle_dict,'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(inv_dict, open(pickle_invdict, 'wb'), pickle.HIGHEST_PROTOCOL)
        print("Done")

    return (multiling_titles, inv_dict) if return_both else inv_dict


# in https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
def simplify_json_file(data_dir, langs, policy="IN_ALL_LANGS", json_file = "latest-all.json.bz2"):
    latest_all_json_file = join(data_dir,json_file)

    if policy not in policies:
        raise ValueError("Policy %s not supported." % policy)

    print("extracting multilingual titles with policy %s (%s)" % (policy,' '.join(langs)))

    lang_prefix = list(langs)
    lang_prefix.sort()
    simple_titles_path = join(data_dir, "extraction_" + "_".join(lang_prefix) + "." + policy)

    def process_entry(last, fo):
        global written
        id = last["id"]
        titles = None
        if policy == "IN_ALL_LANGS" and langs.issubset(last["labels"].keys()):
            titles = {lang: last["labels"][lang]["value"] for lang in langs}
        elif policy == "IN_ANY_LANG":
            titles = {lang: last["labels"][lang]["value"] for lang in langs if lang in last["labels"]}

        if titles:
            fo.write((id+'\t'+'\t'.join([lang+':'+titles[lang] for lang in titles.keys()])+'\n').encode('utf-8'))
            return True
        else:
            return False

    written = 0
    with BZ2File(latest_all_json_file, 'r', buffering=1024*1024*16) as fi, \
            BZ2File(join(data_dir,simple_titles_path+".simple.bz2"),'w') as fo:
        builder = ObjectBuilder()
        completed = 0
        for event, value in ijson.basic_parse(fi, buf_size=1024*1024*16):
             builder.event(event, value)
             if len(builder.value)>1:
                if process_entry(builder.value.pop(0), fo): written += 1
                completed += 1
                print("\rCompleted %d\ttitles %d" % (completed,written), end="")
        print("")

        #process the last entry
        process_entry(builder.value.pop(0))

    return simple_titles_path

"""
Reads all multi-lingual documents in a folder (see wikipedia_tools.py to generate them) and generates, for each of the
specified languages, a list contanining all its documents, so that the i-th element of any list refers to the language-
specific version of the same document. Documents are forced to contain version in all specified languages and to contain
a minimum number of words; otherwise it is discarded.
"""
class MinWordsNotReached(Exception): pass
class WrongDocumentFormat(Exception): pass

def _load_multilang_doc(path, langs, min_words=100):
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import Element, ParseError
    try:
        root = ET.parse(path).getroot()
        doc = {}
        for lang in langs:
            doc_body = root.find('.//doc[@lang="' + lang + '"]')
            if isinstance(doc_body, Element):
                n_words = len(doc_body.text.split(' '))
                if n_words >= min_words:
                    doc[lang] = doc_body.text
                else:
                    raise MinWordsNotReached
            else:
                raise WrongDocumentFormat
    except ParseError:
        raise WrongDocumentFormat
    return doc

#returns the multilingual documents mapped by language, and a counter with the number of documents readed
def fetch_wikipedia_multilingual(wiki_multi_path, langs, min_words=100, deletions=False, max_documents=-1, pickle_name=None):
    if pickle_name and os.path.exists(pickle_name):
        print("unpickling %s" % pickle_name)
        return pickle.load(open(pickle_name, 'rb'))

    multi_docs = list_files(wiki_multi_path)
    mling_documents = {l:[] for l in langs}
    valid_documents = 0
    minwords_exception = 0
    wrongdoc_exception = 0
    for d,multi_doc in enumerate(multi_docs):
        print("\rProcessed %d/%d documents, valid %d/%d, few_words=%d, few_langs=%d" %
              (d, len(multi_docs), valid_documents, len(multi_docs), minwords_exception, wrongdoc_exception),end="")
        doc_path = join(wiki_multi_path, multi_doc)
        try:
            m_doc = _load_multilang_doc(doc_path, langs, min_words)
            valid_documents += 1
            for l in langs:
                mling_documents[l].append(m_doc[l])
        except MinWordsNotReached:
            minwords_exception += 1
            if deletions: os.remove(doc_path)
        except WrongDocumentFormat:
            wrongdoc_exception += 1
            if deletions: os.remove(doc_path)
        if max_documents>0 and valid_documents>=max_documents:
            break

    if pickle_name:
        print("Pickling wikipedia documents object in %s" % pickle_name)
        pickle.dump(mling_documents, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)

    return mling_documents

def random_wiki_sample(l_wiki, max_documents):
    if max_documents == 0: return None
    langs = list(l_wiki.keys())
    assert len(np.unique([len(l_wiki[l]) for l in langs])) == 1, 'documents across languages do not seem to be aligned'
    ndocs_per_lang = len(l_wiki[langs[0]])
    if ndocs_per_lang > max_documents:
        sel = set(np.random.choice(list(range(ndocs_per_lang)), max_documents, replace=False))
        for lang in langs:
            l_wiki[lang] = [d for i, d in enumerate(l_wiki[lang]) if i in sel]
    return l_wiki


if __name__ == "__main__":

    wikipedia_home = "../Datasets/Wikipedia"

    from data.languages import JRC_LANGS_WITH_NLTK_STEMMING as langs
    langs = frozenset(langs)

    simple_titles_path = simplify_json_file(wikipedia_home, langs, policy="IN_ALL_LANGS", json_file="latest-all.json.bz2")
    _, inv_dict = extract_multilingual_titles_from_simplefile(wikipedia_home, simple_titles_path, langs, policy='IN_ALL_LANGS')
    extract_multilingual_documents(inv_dict, langs, join(wikipedia_home,'text'),
                                   out_path=join(wikipedia_home, 'multilingual_docs_JRC_NLTK'))


