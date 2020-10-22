import os
from dataset_builder import MultilingualDataset
from optparse import OptionParser
from util.file import exists
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

parser = OptionParser(usage="usage: %prog datapath [options]")

(op, args) = parser.parse_args()
assert len(args)==1, 'required argument "datapath" missing (path to the pickled dataset)'
dataset = args[0]
assert exists(dataset), 'Unable to find file '+str(dataset)

dataset_file = os.path.basename(dataset)

data = MultilingualDataset.load(dataset)
data.set_view(languages=['it'])
data.show_dimensions()
lXtr, lytr = data.training()
lXte, lyte = data.test()

vect_lXtr = dict()
vectorizer = CountVectorizer()
vect_lXtr['it'] = vectorizer.fit_transform(lXtr['it'])
# print(type(vect_lXtr['it']))

corr = vect_lXtr['it'].T.dot(lytr['it'])
# print(corr.shape)
sum_correlated_class = corr.sum(axis=0)
print(len(sum_correlated_class))
print(sum_correlated_class.max())


w2idx = vectorizer.vocabulary_
idx2w = {v:k for k,v in w2idx.items()}

word_tot_corr = corr.sum(axis=1)
print(word_tot_corr.shape)
dict_word_tot_corr = {v:k for k,v in enumerate(word_tot_corr)}

sorted_word_tot_corr = np.sort(word_tot_corr)
sorted_word_tot_corr = sorted_word_tot_corr[len(sorted_word_tot_corr)-200:]

top_idx = [dict_word_tot_corr[k] for k in sorted_word_tot_corr]
print([idx2w[idx] for idx in top_idx])
print([elem for elem in top_idx])
print(corr[8709])
print('Finished...')