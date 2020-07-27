#!/usr/bin/env bash

dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle
python main_deep_learning.py $dataset --pretrained --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --hidden 256 --tunable --plotmode --test-each 20

python main_deep_learning.py $dataset --supervised --plotmode --test-each 20
python main_deep_learning.py $dataset --supervised --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --supervised --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --supervised --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --supervised --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --supervised --hidden 256 --tunable --plotmode --test-each 20

python main_deep_learning.py $dataset --pretrained --supervised --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --hidden 256 --tunable --plotmode --test-each 20

python main_deep_learning.py $dataset --pretrained --supervised --posteriors --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --posteriors --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --posteriors --hidden 128 --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --posteriors --hidden 128 --tunable --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --posteriors --hidden 256 --plotmode --test-each 20
python main_deep_learning.py $dataset --pretrained --supervised --posteriors --hidden 256 --tunable --plotmode --test-each 20