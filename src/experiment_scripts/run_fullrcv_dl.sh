dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_full_processed.pickle
seeds='1 ' #2 3 4 5' # 6 7 8 9 10'
for seed in $seeds
do
  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_wce.csv --supervised --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_wce_trainable.csv --supervised --tunable --seed $seed
  python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_post_wce_muse_static_plotmode.csv --posteriors --supervised --pretrained --seed $seed --plotmode --test-each 200



  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_muse.csv --pretrained  --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_muse_trainable.csv --pretrained --tunable  --seed $seed

  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_wce_muse.csv --supervised --pretrained  --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_wce_muse_trainable.csv --supervised --pretrained --tunable --seed $seed

#  python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_post_wce_muse_static.csv --posteriors --supervised --pretrained --seed $seed
#  python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_post_wce_muse_trainable_plotmode.csv --posteriors --supervised --pretrained --tunable --seed $seed --plotmode --test-each 200
  #python main_deep_learning.py $dataset --log-file ../log/rcv_fullrun_post_wce_muse_trainable.csv --posteriors --supervised --pretrained --tunable --seed $seed
done