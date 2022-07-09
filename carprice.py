import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
import tensorflow as tf

data = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv',index_col='car_ID')
print(data.head(2))
print(data.info())
print(f"Car Names: {data.CarName.value_counts()}")
print(f"Fuel: {data.fueltype.value_counts()}")
print(f"Door Number: {data.doornumber.value_counts()}")
print(f"Carbody: {data.carbody.value_counts()}")

'''
#PLOTS
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
data.CarName.value_counts().plot(kind='barh',cmap='rainbow')
plt.subplot(2,2,2)
data.fueltype.value_counts().plot(kind='bar',cmap='Spectral')
plt.subplot(2,2,3)
data.doornumber.value_counts().plot(kind='bar',cmap='Spectral')
plt.subplot(2,2,4)
data.carbody.value_counts().plot(kind='bar',cmap='Spectral')


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')


plt.figure(figsize=(10,7))
plt.scatter(data.enginesize,data.price,c='orange')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.title('Correlation')

plt.figure(figsize=(10,7))
plt.scatter(data.horsepower,data.price,c='green')
plt.xlabel('Horse Power')
plt.ylabel('Price')
plt.title('Correlation')

plt.figure(figsize=(10,7))
plt.scatter(data.highwaympg,data.price,c='purple')
plt.xlabel('Highway MPG')
plt.ylabel('Price')
plt.title('Correlation')

plt.figure(figsize=(10,7))
plt.scatter(data.stroke,data.horsepower,c='blue')
plt.xlabel('Stroke')
plt.ylabel('Horsepower')
plt.title('No Correlation')
# plt.show()
'''
#Regresion Lineal
lr = LinearRegression()
lr.fit(data.enginesize.to_numpy().reshape(-1,1),data.price.to_numpy().reshape(-1,1))
y_pred = lr.predict(data.enginesize.to_numpy().reshape(-1,1))
print(lr.coef_)
print(lr.intercept_)
plt.figure(figsize=(10,7))
plt.scatter(data.enginesize.to_numpy(),data.price.to_numpy(),c='blue')
plt.plot(data.enginesize.to_numpy(),y_pred,c='red')

x = np.array(range(5,400))
y = [f for f in (x*168-8005)]
plt.figure(figsize=(10,7))
plt.scatter(data.enginesize.to_numpy(),data.price.to_numpy(),c='blue')
plt.plot(x,y,c='orange')

#Create a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])
model.summary()
model.fit(data.enginesize.to_numpy(),data.price.to_numpy(),epochs=50)
y_pred = model.predict(x)

print(model.weights)
plt.figure(figsize=(10,7))
plt.scatter(data.enginesize.to_numpy(),data.price.to_numpy(),c='blue')
plt.plot(x,y_pred,c='green')
plt.show()


