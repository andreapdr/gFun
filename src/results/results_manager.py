import pandas as pd
import numpy as np

df = pd.read_csv("/home/andreapdr/funneling_pdr/src/results/results.csv", delimiter='\t')
pivot = pd.pivot_table(df, values=['time', 'macrof1', 'microf1', 'macrok', 'microk'], index=['embed'], aggfunc=[np.mean, np.std])
print(pivot)
print('Finished ...')