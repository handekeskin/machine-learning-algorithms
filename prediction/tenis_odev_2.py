#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#2.1 veri yükleme
data = pd.read_csv('../data/tennis_data.csv')

#encoder ordinal , nominal -> numeric (kategorik dataları numeric datalara çevirme)
data2 = data.apply(LabelEncoder().fit_transform)

outlook = data2.iloc[:,0:1].values
ohe = OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

#numpy dizileri dataframe dönüşümü ve data birleştirme
sonuc1 = pd.DataFrame(data = outlook , index=range(14), columns=['overcast','rainy','sunny'])#array'ı dataframe'e çevirdi
print(sonuc1)

sonuc = pd.concat ([data2.iloc[:,-2:], sonuc1, data.iloc[:,1:3]],axis=1)
print(sonuc)

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(sonuc.iloc[:,:-1],sonuc.iloc[:,-1:],test_size=0.33,random_state=0)

#çoklu lineer regresyon modeli-cinsiyet için-
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_prediction =  mlr.predict(x_test)

print(mlr_prediction)
print(y_test)

#backward selection

import statsmodels.formula.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=sonuc.iloc[:,-1:], axis=1 )
X_l = sonuc.iloc[:,[0,1,2,3,4,5]].values
print(X_l)
r_ols = sm.OLS(endog=sonuc.iloc[:,-1:],exog=X_l)
r = r_ols.fit()
print(r.summary())

sonuc = sonuc.iloc[:,1:]

X = np.append(arr=np.ones((14,1)).astype(int), values=sonuc.iloc[:,-1:], axis=1 )
X_l = sonuc.iloc[:,[0,1,2,3,4]].values
print(X_l)
r_ols = sm.OLS(endog=sonuc.iloc[:,-1:],exog=X_l)
r = r_ols.fit()
print(r.summary())

x_train= x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
mlr.fit(x_train,y_train)
mlr_prediction_2=mlr.predict(x_test)

print(y_test)
print(mlr_prediction_2)


