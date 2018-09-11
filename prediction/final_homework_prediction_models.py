#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.metrics import r2_score

#2.1 veri yükleme
data = pd.read_csv('../data/salary_extended.csv')

#her değişkenin birbiri arasındaki koralesyonu pandaki bir fonksiyon üzerinden hesaplanabilir.
#bu funksiyon bağımsız değişkenler arasındaki ilişkiyide veriyor.
print(data.corr())

#dataframe dilimleme(slice)
x=data.iloc[:,2:5]
y=data.iloc[:,5:]
X=x.values
Y=y.values
"""
#lineer regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#backward selection
import statsmodels.formula.api as sm
r_ols = sm.OLS(lin_reg.predict(X),X)
r = r_ols.fit()
print(r.summary())

"""
#dataframe dilimleme(slice)
x1=data.iloc[:,2:3]
y1=data.iloc[:,5:]
X1=x1.values
Y1=y.values

#lineer regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X1,Y1)

#backward selection
print("Linear OLS")
r_ols = sm.OLS(lin_reg.predict(X1),X1)
r = r_ols.fit()
print(r.summary())

#lineer regression r square evaluation
print("evaluation of lineer regression:")
print(r2_score(Y1,lin_reg.predict(X1)))
"""
#polynomial-regression (2. degree)
from sklearn.preprocessing import PolynomialFeatures
pl_1=PolynomialFeatures(degree=2)
pl_X=pl_1.fit_transform(X)
lr_2=LinearRegression()
lr_2.fit(pl_X,Y)

#backward selection
r_ols = sm.OLS(lr_2.predict(pl_1.fit_transform(X)),X)
r = r_ols.fit()
print(r.summary())
"""
#polynomial-regression (2. degree)
from sklearn.preprocessing import PolynomialFeatures
pl_1=PolynomialFeatures(degree=2)
pl_X=pl_1.fit_transform(X1)
lr_2=LinearRegression()
lr_2.fit(pl_X,Y1)

#backward selection
print("Linear polynomial-regression (2 degree) OLS")
r_ols = sm.OLS(lr_2.predict(pl_1.fit_transform(X1)),X1)
r = r_ols.fit()
print(r.summary())

#r2 score evaluation in polynomial-regression (2. degree)
print("evaluation of polynomial-regression (2. degree):")
print(r2_score(Y1,lr_2.predict(pl_1.fit_transform(X1))))
"""
#polynomial-regression (4. degree)
pl_2=PolynomialFeatures(degree=4)
pl_X2=pl_2.fit_transform(X)
lr_3=LinearRegression()
lr_3.fit(pl_X2,Y)

#backward selection
r_ols = sm.OLS(lr_3.predict(pl_2.fit_transform(X)),X)
r = r_ols.fit()
print(r.summary())
"""
#polynomial-regression (4. degree)
pl_2=PolynomialFeatures(degree=4)
pl_X2=pl_2.fit_transform(X1)
lr_3=LinearRegression()
lr_3.fit(pl_X2,Y1)

#backward selection
print("Linear polynomial-regression (4 degree) OLS")
r_ols = sm.OLS(lr_3.predict(pl_2.fit_transform(X1)),X1)
r = r_ols.fit()
print(r.summary())

#r2 score evaluation in polynomial-regression (4. degree)
print("evaluation of polynomial-regression (4. degree):")
print(r2_score(Y1,lr_3.predict(pl_2.fit_transform(X1))))

#verilerin ölçeklendirilmesi(scale)
from sklearn.preprocessing import StandardScaler
st1 = StandardScaler()
X_scale=st1.fit_transform(X)
st2 = StandardScaler()
Y_scale=st2.fit_transform(Y)
"""
#SVR model for rbf
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X_scale,Y_scale)

#backward selection
r_ols = sm.OLS(svr.predict(X_scale),X_scale)
r = r_ols.fit()
print(r.summary())
"""
from sklearn.preprocessing import StandardScaler
st1 = StandardScaler()
X1_scale=st1.fit_transform(X1)
st2 = StandardScaler()
Y1_scale=st2.fit_transform(Y1)

#SVR model for rbf
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X1_scale,Y1_scale)

#backward selection
print("SVR OLS")
r_ols = sm.OLS(svr.predict(X1_scale),X1_scale)
r = r_ols.fit()
print(r.summary())

#r2 score evaluation in SVR
print("evaluation of SVR:")
print(r2_score(Y1_scale,svr.predict(X1_scale)))
"""
#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

#backward selection
r_ols = sm.OLS(r_dt.predict(X),X)
r = r_ols.fit()
print(r.summary())
"""
#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X1,Y1)

#backward selection
print("Decision Tree OLS")
r_ols = sm.OLS(r_dt.predict(X1),X1)
r = r_ols.fit()
print(r.summary())

#r2 score evaluation in Decision Tree
print("evaluation of decision tree:")
print(r2_score(Y1,r_dt.predict(X1)))
"""
#random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,Y)

#backward selection
r_ols = sm.OLS(rf.predict(X),X)
r = r_ols.fit()
print(r.summary())
"""
#random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X1,Y1)

#backward selection
print("random-forest OLS")
r_ols = sm.OLS(rf.predict(X1),X1)
r = r_ols.fit()
print(r.summary())

#r2 score evaluation in random-forest
print("evaluation of random forest:")
print(r2_score(Y1,rf.predict(X1)))

#lineer regression r square evaluation
print("evaluation of lineer regression:")
print(r2_score(Y1,lin_reg.predict(X1)))

#r2 score evaluation in polynomial-regression (2. degree)
print("evaluation of polynomial-regression (2. degree):")
print(r2_score(Y1,lr_2.predict(pl_1.fit_transform(X1))))

#r2 score evaluation in polynomial-regression (4. degree)
print("evaluation of polynomial-regression (4. degree):")
print(r2_score(Y1,lr_3.predict(pl_2.fit_transform(X1))))

#r2 score evaluation in SVR
print("evaluation of SVR:")
print(r2_score(Y1_scale,svr.predict(X1_scale)))

#r2 score evaluation in Decision Tree
print("evaluation of decision tree:")
print(r2_score(Y1,r_dt.predict(X1)))

#r2 score evaluation in random-forest
print("evaluation of random forest:")
print(r2_score(Y1,rf.predict(X1)))
