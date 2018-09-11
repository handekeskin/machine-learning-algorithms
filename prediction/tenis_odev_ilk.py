#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#2.1 veri yükleme
data = pd.read_csv('../data/tennis_data.csv')

print(data)

#encoder ordinal , nominal -> numeric (kategorik dataları numeric datalara çevirme)
from sklearn.preprocessing import LabelEncoder
outlook = data.iloc[:,0:1].values
le = LabelEncoder()
outlook[:,0]=le.fit_transform(outlook[:,0])
print(outlook)

from sklearn.preprocessing import LabelEncoder
windy = data.iloc[:,3:4].values
print(windy)
le = LabelEncoder()
windy[:,0]=le.fit_transform(windy[:,0])
print(windy)

from sklearn.preprocessing import LabelEncoder
play = data.iloc[:,-1:].values
le = LabelEncoder()
play[:,0]=le.fit_transform(play[:,0])
print(play)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

ohe = OneHotEncoder(categorical_features='all')
windy=ohe.fit_transform(windy).toarray()
print(windy)

#numpy dizileri dataframe dönüşümü
sonuc1 = pd.DataFrame(data = outlook , index=range(14), columns=['overcast','rainy','sunny'])#array'ı dataframe'e çevirdi
print(sonuc1)

temprature=data.iloc[:,1:2].values
sonuc2 = pd.DataFrame(data = temprature, index=range(14), columns=['temprature'])
print(sonuc2)

sonuc3= pd.DataFrame(data = windy[:,-1:], index=range(14), columns=['windy'])
print(sonuc3)

sonuc4= pd.DataFrame(data = play, index=range(14), columns=['play'])
print(sonuc4)

huminity=data.iloc[:,2:3].values
sonuc5 = pd.DataFrame(data = huminity, index=range(14), columns=['huminity'])
print(sonuc5)

#data birleştirme
s1=pd.concat([sonuc1,sonuc2,sonuc3,sonuc4],axis=1)
print(s1)
s2=pd.concat([s1,sonuc5],axis=1)
print(s2)

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(s1,sonuc5,test_size=0.33,random_state=0)

#çoklu lineer regresyon modeli-
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_prediction =  mlr.predict(x_test)

#encoder ordinal , nominal -> numeric (kategorik dataları numeric datalara çevirme)
data2 = data.apply(LabelEncoder().fit_transform)

#numpy dizileri dataframe dönüşümü ve data birleştirme
sonuc6 = pd.DataFrame(data = outlook , index=range(14), columns=['overcast','rainy','sunny'])#array'ı dataframe'e çevirdi
print(sonuc1)

sonuc = pd.concat ([data2.iloc[:,-2:], sonuc6, data.iloc[:,1:3]],axis=1)
print(sonuc)
print(s2)

#backward selection

import statsmodels.formula.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=s2.iloc[:,:-1], axis=1 )
X_l = s2.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog=sonuc.iloc[:,-1:].astype(float),exog=X_l.astype(float))
r = r_ols.fit()
print(r.summary())

X = np.append(arr=np.ones((14,1)).astype(int), values=s2.iloc[:,:-1], axis=1 )
X_l = s2.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog=sonuc.iloc[:,-1:].astype(float),exog=X_l.astype(float))
r = r_ols.fit()
print(r.summary())

s1 = s2.iloc [:,[0,1,2,3,5]]
print(s1)
#tekrar prediction yaparsak
#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(s1,sonuc5,test_size=0.33,random_state=0)

#çoklu lineer regresyon modeli-
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_prediction =  mlr.predict(x_test)

print(y_test)
print(mlr_prediction)

