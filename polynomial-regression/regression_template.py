#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#2.1 veri yükleme(data load)
salary_data = pd.read_csv('../data/salary.csv')

#dataframe dilimleme(slice)
education_level=salary_data.iloc[:,1:2]
salary=salary_data.iloc[:,2:]

#numpy array dönüşümü
education_level_array=education_level.values
salary_array=salary.values

#lineer regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(education_level_array,salary_array)
#lineer regression r square evaluation
print("evaluation of lineer regression:")
print(r2_score(salary,lr.predict(education_level_array)))


#polynomial-regression (2. degree)
from sklearn.preprocessing import PolynomialFeatures
pl_1=PolynomialFeatures(degree=2)
pl_education_level=pl_1.fit_transform(education_level_array)
lr_2=LinearRegression()
lr_2.fit(pl_education_level,salary)
#r2 score evaluation in polynomial-regression (2. degree)
print("evaluation of polynomial-regression (2. degree):")
print(r2_score(salary,lr_2.predict(pl_1.fit_transform(education_level_array))))

#polynomial-regression (4. degree)
pl_2=PolynomialFeatures(degree=4)
pl_education_level=pl_2.fit_transform(education_level_array)
lr_3=LinearRegression()
lr_3.fit(pl_education_level,salary)
#r2 score evaluation in polynomial-regression (4. degree)
from sklearn.metrics import r2_score
print("evaluation of polynomial-regression (4. degree):")
print(r2_score(salary,lr_3.predict(pl_2.fit_transform(education_level_array))))

#visualisation

#lineer regression visualisation
plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level,lr.predict(education_level_array),color='blue')
plt.title('linear regressor')
plt.show()
#polynomial-regression visualisation (2. degree)
plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,lr_2.predict(pl_1.fit_transform(education_level_array)),color='blue')
plt.title('polynomial regressor (2. degree)')
plt.show()
#polynomial-regression visualisation
plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,lr_3.predict(pl_2.fit_transform(education_level_array)),color='blue')
plt.title('polynomial regressor (4. degree)')
plt.show()

#predition
#for lineer
print(lr.predict(11))
print(lr.predict(6.6))
#for 2. degree polinomial regression
print(lr_2.predict(pl_1.fit_transform(11)))
print(lr_2.predict(pl_1.fit_transform(6.6)))
#for 4. degree polinomial regression
print(lr_3.predict(pl_2.fit_transform(11)))
print(lr_3.predict(pl_2.fit_transform(6.6)))

#verilerin ölçeklendirilmesi(scale)
from sklearn.preprocessing import StandardScaler
st1 = StandardScaler()
X_scale=st1.fit_transform(education_level_array)
st2 = StandardScaler()
Y_scale=st2.fit_transform(salary_array)

#SVR model for rbf
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X_scale,Y_scale)
#r2 score evaluation in SVR
print("evaluation of SVR:")
print(r2_score(Y_scale,svr.predict(X_scale)))

plt.scatter(X_scale,Y_scale,color='red')
plt.plot(X_scale,svr.predict(X_scale),color='blue')
plt.title('rbf')
plt.show()


#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(education_level_array,salary_array)
#r2 score evaluation in Decision Tree
print("evaluation of decision regressor:")
print(r2_score(salary_array,r_dt.predict(education_level_array)))

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,r_dt.predict(education_level_array),color='blue')
plt.title('Desicion Tree Regressor')
plt.show()

print(r_dt.predict(11))
print(r_dt.predict(6.6))

#random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(education_level_array,salary_array)
#r2 score evaluation in random-forest
print("evaluation of random forest:")
print(r2_score(salary_array,rf.predict(education_level_array)))

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,rf.predict(education_level_array),color='blue')
plt.title('random-forest Regressor')
plt.show()

print(rf.predict(11))
print(rf.predict(6.6))