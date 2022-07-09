from email import message
import os
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score,recall_score
import tensorflow as tf

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
print(data.head(2))
print(data.columns.tolist())
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
print(data.columns.tolist())
print(data.head(3))

#PLOTS
sns.displot(data['class'].tolist())
#plt.show()

data['class'] = data['class'].map({'ham':0,'spam':1})
y = data['class'].to_numpy()
x = data['message'].to_numpy()


#Split dataset into train,test sets (for validation sets have to do twice)
# xtrain,xtest,ytrain,ytest = train_test_split(data['message'].to_numpy(),data['class'].to_numpy(),test_size=.2,random_state=42,stratify=data['class'].to_numpy())
xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=.2, random_state=42, stratify = y)
print(f"Train shapes: {xtrain.shape}, {ytrain.shape}")
print(f"Test shapes: {xtest.shape}, {ytest.shape}")
print(xtrain[0],ytrain[0])
print(type(xtrain))

#Create Naive Bayes Multinomial Model for clasification
model_base = Pipeline([
    ("cv",CountVectorizer()),
    ("tfid",TfidfTransformer()),
    ("clf",MultinomialNB())
])
print(model_base)
model_base.fit(xtrain,ytrain)
print(model_base.score(xtrain,ytrain))
ypred = model_base.predict(xtest)
print("Precision: ",precision_score(ytest,ypred))
print("Recall: ",recall_score(ytest,ypred))

# sample = input('Enter a message: ')
# data = cv.transform([sample]).toarray()
# print(clf.predict(data))

train_dataset = tf.data.Dataset.from_tensor_slices((xtrain,ytrain))
test_dataset = tf.data.Dataset.from_tensor_slices((xtest,ytest))

model_nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,),dtype=tf.string),
    tf.keras.layers.experimental.preprocessing.TextVectorization(),
    tf.keras.layers.Embedding(input_dim=len((tf.keras.layers.experimental.preprocessing.TextVectorization().adapt(xtrain)).get_vocabulary()),output_dim=128),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(60,return_sequences=True),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model_nn.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
model_nn.summary()
model_nn.fit(train_dataset,
          epochs=1,
          validation_data=test_dataset)
