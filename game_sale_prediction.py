import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('C:\Code\Machine_Learning\Data_Science_projects\Video_Games_Sales_as_at_22_Dec_2016.csv')
print(df.head())
print(df.info())
df_num = df.drop(columns=['Name','Platform','Genre','Publisher','User_Score',
                    'Developer','Rating'])
print(df_num.info())

#plt.figure()
# sns.heatmap(df_num.corr(),cmap='coolwarm',annot=True)
# sns.pairplot(data=df,hue='Genre',vars=['NA_Sales','EU_Sales','JP_Sales','Global_Sales'])
# sns.barplot(data=df,x='Genre',y='Global_Sales')
# plt.figure()
# plt.bar(df['Year_of_Release'],df['Global_Sales'])
# plt.xlim(0,5)
# plt.show()

print(df.isna().sum())
data = df.drop(columns=['Name','Year_of_Release','Publisher',
                    'Critic_Score','Critic_Count',
                    'User_Score','User_Count',
                    'Developer','Rating'])
print(data.isnull().sum())
print(data['Genre'].value_counts(True))
data['Genre'] = data['Genre'].fillna(value='Action')
# sns.histplot(data=data,x='Genre')
# plt.show()
print(data.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
from sklearn.decomposition import SparseCoder
from sklearn.compose import make_column_transformer

x = data.drop(columns=['Global_Sales','Platform'],axis=1)
y = data['Global_Sales']

ct = make_column_transformer((MinMaxScaler(),['NA_Sales','EU_Sales','JP_Sales','Other_Sales']),
                            (OneHotEncoder(handle_unknown='ignore'),['Genre']))
#ct.fit(x)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=42)
#ct.fit(xtrain)
xtrain_transformed = ct.fit_transform(xtrain)
# ct.fit(xtest)
xtest_transformed = ct.fit_transform(xtest)
print(f"Train transf: {xtrain_transformed.shape}\n Test trans: {xtest_transformed.shape}")

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain_transformed,ytrain)
print("Train score",rf.score(xtrain_transformed,ytrain))
print("Test score",rf.score(xtest_transformed,ytest))

ct = MinMaxScaler()
x = data.drop(columns=['Global_Sales','Platform','Genre'],axis=1)
y = data['Global_Sales']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=42)
xtrain_transformed = ct.fit_transform(xtrain)
xtest_transformed = ct.fit_transform(xtest)
print(f"Train: {xtrain_transformed.shape}\n Test: {xtest_transformed.shape}")

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain_transformed,ytrain)
print(f"Linear score: {lr.score(xtrain_transformed,ytrain)}")
print(f"Linear score: {lr.score(xtest_transformed,ytest)}")