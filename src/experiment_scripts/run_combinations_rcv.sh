#!/usr/bin/env bash

dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle
logfile=./results/final_combinations_rcv.csv
#A.2: ensembling feature sets (combinations of posteriors, wce, muse):
#	- exploring different ways of putting different feature sets together: concatenation, FeatureSetToPosteriors, averaging, voting, etc...
#		(no one seems to improve over standard funnelling [the improved version after A.1] with posteriors probabilities...)

# aggregation=concatenation
#python main_gFun.py $dataset -o $logfile -P -U -r -z --l2
#python main_gFun.py $dataset -o $logfile -P -S -r -z --l2
#python main_gFun.py $dataset -o $logfile -U -S -r -z --l2
#python main_gFun.py $dataset -o $logfile -P -U -S -r -z --l2
#
##FeatureSetToPosteriors (aggregation mean)
python main_multimodal_cls.py $dataset -o $logfile -P -U -r -a -z --l2 --allprob
python main_multimodal_cls.py $dataset -o $logfile -P -S -r -a -z --l2 --allprob
python main_multimodal_cls.py $dataset -o $logfile -U -S -r -a -z --l2 --allprob
python main_multimodal_cls.py $dataset -o $logfile -P -U -S -r -a -z --l2 --allprob

##FeatureSetToPosteriors
#python main_gFun.py $dataset -o $logfile -P -U -r -z --l2 --allprob
#python main_gFun.py $dataset -o $logfile -P -S -r -z --l2 --allprob
#python main_gFun.py $dataset -o $logfile -U -S -r -z --l2 --allprob
#python main_gFun.py $dataset -o $logfile -P -U -S -r -z --l2 --allprob

#MajorityVoting
#python main_majorityvoting_cls.py $dataset -o $logfile -P -U -r
#python main_majorityvoting_cls.py $dataset -o $logfile -P -S -r
#python main_majorityvoting_cls.py $dataset -o $logfile -U -S -r
#python main_majorityvoting_cls.py $dataset -o $logfile -P -U -S -r