#!/usr/bin/env bash

logfile=../log/log_pre_jrc.csv
dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run0.pickle
python main_deep_learning.py $dataset --log-file $logfile --pretrained --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --hidden 256 --tunable --plotmode --test-each 20

python main_deep_learning.py $dataset --log-file $logfile --supervised --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --supervised --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --supervised --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --supervised --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --supervised --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --supervised --hidden 256 --tunable --plotmode --test-each 20

python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --hidden 256 --tunable --plotmode --test-each 20

python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --hidden 256 --tunable --plotmode --test-each 20