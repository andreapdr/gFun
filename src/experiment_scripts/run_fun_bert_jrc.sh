#!/usr/bin/env bash

#dataset_path=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run
#logfile=../log/log_FunBert_jrc.csv
#
#runs='0 1 2 3 4'
#for run in $runs
#do
#  dataset=$dataset_path$run.pickle
#  python main_deep_learning.py $dataset --supervised --pretrained --posteriors --mbert --log-file $logfile #--tunable
#done

dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle
logfile=../log/log_FunBert_fulljrc_static.csv

python main_deep_learning.py $dataset --supervised --pretrained --posteriors --mbert --log-file $logfile