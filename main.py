import scipy.io
import pandas as pd

mat = scipy.io.loadmat('Xtrain.mat')
# print(mat['Xtrain'].shape)

df = pd.DataFrame(mat['Xtrain'])
print(df.head())
print(df.shape)

df.to_csv('Xtrain.csv', index=False)
