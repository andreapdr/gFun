#!/usr/bin/env bash

dataset_path=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run
logfile=../log/log10run_dl_jrc.csv

runs='0 1 2 3 4 5 6 7 8 9'
for run in $runs
do
  dataset=$dataset_path$run.pickle
  python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --tunable --plotmode --test-each 20
done