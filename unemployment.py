import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/unemployment.csv",index_col='Date')
print(f"Columns names: {data.columns}")
print(f"Info dataframe: {data.info()}")
print(f"Index: {data.index}")

columnas = data.columns.tolist()
print(columnas[3])
plt.figure()
data[columnas[3]].plot(kind='line')
plt.figure()
sns.heatmap(data.corr())
plt.figure()
sns.distplot(data[columnas[2]])
plt.figure()
sns.histplot(data[columnas[3]])

print(type(columnas[3]),columnas[3],columnas[0])

pd.pivot_table(data,index=['Region'],values=columnas[3]).plot(kind='barh')
plt.show()