dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle
seeds='5' #2 3 4 5 6 7 8 9 10'
for seed in $seeds
do
  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_wce.csv --supervised --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_wce_trainable.csv --supervised --tunable --seed $seed
  python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_post_wce_muse_static.csv --posteriors --supervised --pretrained --seed $seed  --force

  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_muse.csv --pretrained  --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_muse_trainable.csv --pretrained --tunable  --seed $seed

  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_wce_muse.csv --supervised --pretrained  --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_wce_muse_trainable40000.csv --supervised --pretrained --tunable --seed $seed
  #python main_deep_learning.py $dataset --log-file ../log/jrc_fullrun_post_wce_muse_trainable.csv --posteriors --supervised --pretrained --tunable --seed $seed  --force

done