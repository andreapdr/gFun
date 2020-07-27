dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_full_processed.pickle
seeds='1 2 3 4 5 6 7 8 9 10'
for seed in $seeds
do
  python main_deep_learning.py $dataset --log-file ../log/time_GRU.csv --supervised  --nepochs 50 --seed $seed
  done