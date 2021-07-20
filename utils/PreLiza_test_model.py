#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



df = pd.read_csv("../assets/houses.csv")
df
# cleaning from Hakan

# removes about 1500 duplicates
df = df.drop_duplicates(subset=['location', 'price', 'area', 'building_condition'])
df

# removing any row that contains empty area information because it's a key feature
df = df.dropna(subset=['area', 'facade_count', 'price'])
#df.price.sort_values()
#print(df[df.price.isnull()])



Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
ndf = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
#ndf.price.sort_values()

#ndf



area = ndf.area.to_numpy().reshape(-1,1)
area.shape
price = ndf.price.to_numpy().reshape(-1,1)
price.shape
X_train, X_test, y_train, y_test = train_test_split(area, price, test_size=0.3)


reg = LinearRegression().fit(X_train,y_train)


plt.figure(figsize = (15,5)) 

plt.scatter(ndf.area,ndf.price,c='green', alpha=0.3) 

plt.plot(area,reg.predict(area), c='black') 

plt.ticklabel_format(style="plain")
plt.xlabel('area') 
plt.ylabel('price') 
#plt.ylim(ymax=3500000,ymin=10)
#plt.xlim(xmax=1100,xmin=10)


reg.score(X_test,y_test)
#0.22


location = ndf.location.to_numpy().reshape(-1, 1)
location.shape


X_train, X_test, y_train, y_test = train_test_split(location, price, test_size=0.3)
reg = LinearRegression().fit(X_train,y_train)


plt.figure(figsize = (15,5)) 

plt.scatter(ndf.location,ndf.price,c='green', alpha=0.3) 

plt.plot(location,reg.predict(location), c='black') 

plt.ticklabel_format(style="plain")
plt.xlabel('location') 
plt.ylabel('price') 
#plt.ylim(ymax=3500000,ymin=10)
#plt.xlim(xmax=1100,xmin=10)


reg.score(X_test,y_test)


Brussels = ndf.query("999 < location < 1300")
Brussels



br_area = Brussels.area.to_numpy().reshape(-1,1)
br_price = Brussels.price.to_numpy().reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(br_area, br_price, test_size=0.3)


reg = LinearRegression().fit(X_train,y_train)

plt.figure(figsize = (15,5)) 

plt.scatter(br_area,br_price,c='green', alpha=0.3) 

plt.plot(br_area,reg.predict(br_area), c='black') 

plt.ticklabel_format(style="plain")
plt.xlabel('area') 
plt.ylabel('price') 
#plt.ylim(ymax=3500000,ymin=10)
#plt.xlim(xmax=1100,xmin=10)


reg.score(X_test,y_test)
