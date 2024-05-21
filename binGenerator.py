import pandas as pd
import numpy as np

df = pd.read_csv('new_dataset_corrected - Binned.csv')



bins = [-1, 25, 100, np.inf]
names = ['<25', '25--100', '100+']

df['internal_links_new'] = pd.cut(df['internal_links'], bins, labels=names)

print(df)

df.to_csv('new_dataset_corrected - Binned.csv', index=False) 