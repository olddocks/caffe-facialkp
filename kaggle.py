import pandas as pd
import numpy as np
import os.path, time

'''
df = pd.read_csv('fkp_output.csv',header=0)

df = df.unstack()

df.to_csv('output.csv')

'''

a = pd.read_csv('output.csv',header=0)
b = pd.read_csv('IdLookupTable.csv',header=0)

a.ImageId += 1

#a = pd.DataFrame(df)
#a.columns = ['ImageId','FeatureName','Location']

b = b.drop('Location',axis=1)
merged = b.merge(a, on=['ImageId','FeatureName'] )

merged.to_csv('kaggle.csv', index=0,cols = ['RowId','Location'] )

