import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/Billionaire.csv")

print(data.head())
print()
print(data.info)
print()
print(data.describe())
print()
print(data.columns)
print()
print(data.isna().sum())
print()
age_mean = data.Age.mean().astype('int')
data.Age = data.Age.fillna(value=age_mean)
print(data.Age.isna().sum())
print(data[["Country","Source","Industry"]].nunique())
print(data.Country.unique())
print("Networth : ",data.NetWorth.describe())

### PREPROCESSING NETWORTH ###
data.NetWorth = data.NetWorth.astype('string')
print(data.NetWorth.dtype)

data.NetWorth = data.NetWorth.str.replace('$','')
data.NetWorth = data.NetWorth.str.replace('B','')
data.NetWorth = data.NetWorth.astype('float')
print(data.NetWorth.value_counts())


### PLOTS / VIEWS ###
# Country/Industry/Age
a = data.Country.value_counts()[:5]
b = data.Industry.value_counts()[:5]
c = data.Age
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)#(nrows,ncols,index)
plt.pie(a.values,labels=a.index)
plt.title("Top 5 Countries with Most Billionaires")
plt.subplot(2,2,2)
plt.pie(b.values,labels=b.index)
plt.title("Top 5 Industry with Most Billionaires")
plt.subplot(2,2,3)
plt.hist(c)
plt.title("Histogram of Billionaires'age")
plt.subplot(2,2,4)
plt.hist(data.NetWorth.value_counts(),bins=10)
plt.title("Histogram of Billionaires'NetWorth")
plt.show()
#Sort by names
plt.figure(figsize=(20, 10))
sns.histplot(x="Name", hue="NetWorth", data=data)
plt.show()
