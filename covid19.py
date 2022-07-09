import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px

data = pd.read_csv(filepath_or_buffer='C:\Code\Machine_Learning\Data_Science_projects\country_wise_latest.csv')
print(data.head())
print(data.tail())
print(data.info())
print(data.isna().sum())
print(data.columns)
print("----",max(data['Country/Region'].value_counts()))
print("----",data['WHO Region'].value_counts())
cols = data.columns.tolist()
data.pop(cols[0])
data.pop(cols[-1])
#sns.heatmap(data.corr(),cmap="coolwarm",annot=True,cbar=False)

#data['WHO Region'].plot()
plt.show()