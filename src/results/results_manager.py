import pandas as pd
import numpy as np

# df = pd.read_csv("/home/andreapdr/funneling_pdr/src/results/final_results.csv", delimiter='\t')
df = pd.read_csv("10run_rcv_final_results.csv", delimiter='\t')
pivot = pd.pivot_table(df, values=['macrof1', 'microf1', 'macrok', 'microk'], index=['method', 'id', 'optimp', 'zscore', 'l2', 'wescaler', 'pca', 'sif'], aggfunc=[np.mean, np.std])
with pd.option_context('display.max_rows', None):
    print(pivot.round(3))
print('Finished ...')


