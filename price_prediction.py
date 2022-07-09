import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('C:\Code\Machine_Learning\Data_Science_projects\MSFT.csv',index_col='Date')
data = data.rename(columns={"Close":"Price"})
print(data.head(3))
print(data.isna().sum())

sns.heatmap(data.corr(),cmap='coolwarm',annot=True)
# plt.show()

x = data[["Open","High","Low"]].to_numpy()
y = data.Price.to_numpy()
y = y

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape,y_train.shape)

plt.figure()
plt.subplot(3,2,1)
plt.scatter(x_train[:,0],y_train,c='blue',label='Train')
plt.title('Train set')
plt.legend()

print("-------------------------")
print("Random Forest Regressor")
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
print(rf.fit(x_train,y_train))
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
print(f"Best params: {rf.estimator_params},{rf.base_estimator_}")
pred_rf = rf.predict(x_test)
plt.subplot(3,2,2)
plt.scatter(x_train[:,0],y_train,c='blue',s=3,label='Train')
plt.scatter(x_test[:,0],pred_rf,c='orange',s=5,label='Test')
plt.title('Random Forest Regressor')
plt.legend()
print("-------------------------")

print("Decision Trees")
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print(dtr.score(x_train,y_train))
print(cross_val_score(dtr, x_test, y_test, cv=10))
print("Tree depth",dtr.get_depth())
print("Tree leaves",dtr.get_n_leaves())
print("Tree params",dtr.get_params())
pred_dtr = dtr.predict(x_test)
plt.subplot(3,2,3)
plt.scatter(x_train[:,0],y_train,c='blue',s=3,label='Train')
plt.scatter(x_test[:,0],pred_dtr,c='green',s=5,label='Test')
plt.title('Decision Tree Regressor')
plt.legend()
print("-------------------------")

print("Grid Search Decision Trees")
from sklearn.model_selection import GridSearchCV
tree = DecisionTreeRegressor()
param_grid = {'max_depth':[5,10,15,20,25]}
grid = GridSearchCV(tree,param_grid=param_grid)
# print("Grid params: ",grid.estimator.get_params().keys())
grid.fit(x_train,y_train)
print("Best params: ",grid.best_params_,grid.best_estimator_)
print("Grid Score: ",grid.score(x_test,y_test))
pred_grid = grid.predict(x_test)
plt.subplot(3,2,4)
plt.scatter(x_train[:,0],y_train,c='blue',s=3,label='Train')
plt.scatter(x_test[:,0],pred_grid,c='red',s=5,label='Test')
plt.title('Grid Tree Regressor')
plt.legend()
print("-------------------------")

print("Baggin Ensemble Method Decision Trees")
from sklearn.ensemble import BaggingRegressor
tree = DecisionTreeRegressor()
bag = BaggingRegressor(tree,n_estimators=100,max_samples=0.8,random_state=1)
bag.fit(x_train,y_train)
print("Bag Score: ",bag.score(x_test,y_test))
print("Best params: ",bag.get_params)
pred_bag = bag.predict(x_test)
plt.subplot(3,2,5)
plt.scatter(x_train[:,0],y_train,c='blue',s=3,label="Train")
plt.scatter(x_test[:,0],pred_bag,c='purple',s=5,label='Test')
plt.title('Bagging Tree Regressor')
plt.legend()
print("-------------------------")

print("Grid Search Random Forest parameters")
rf = RandomForestRegressor()
param_grid = {"max_depth":[3,5,10,15,20,25,30,35,40],"max_features":["auto", "sqrt", "log2"]}
grid = GridSearchCV(rf,param_grid=param_grid)
print(f"RF params: {grid.estimator.get_params().keys()}")
grid.fit(x_train,y_train)
print("Grid Score: ",grid.score(x_test,y_test))
print("Grid RF Best params: ",grid.best_estimator_)
pred_grid = grid.predict(x_test)
plt.subplot(3,2,6)
plt.scatter(x_train[:,0],y_train,c='blue',s=3,label='Train')
plt.scatter(x_test[:,0],pred_grid,c='gold',s=5,label='Test')
plt.title('Grid Random Regressor')
print("-------------------------")
plt.legend()
plt.show()