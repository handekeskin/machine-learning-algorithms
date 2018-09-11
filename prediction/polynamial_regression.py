#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 veri yükleme
salary_data = pd.read_csv('../data/salary.csv')

#polinom regresyonda formül y=a+bX1+cX2+dX1^2+eX2^2+fX1X2 şekilde yada y=a+bX+cX^2+dX^3+eX^4+fX^5.. şeklindedir. Yani polinom denklemidir.

education_level=salary_data.iloc[:,1:2]
salary=salary_data.iloc[:,2:]

education_level_array=education_level.values
salary_array=salary.values

#doğrusal regresyon
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(education_level_array,salary_array)

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level,lr.predict(education_level_array),color='blue')
plt.show()

#polynomial-regression (2. degree)
from sklearn.preprocessing import PolynomialFeatures
pl_1=PolynomialFeatures(degree=2)
pl_education_level=pl_1.fit_transform(education_level_array)
lr_2=LinearRegression()
lr_2.fit(pl_education_level,salary)

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,lr_2.predict(pl_1.fit_transform(education_level_array)),color='blue')
plt.show()

#polynomial-regression (4. degree)
from sklearn.preprocessing import PolynomialFeatures
pl_1=PolynomialFeatures(degree=4)
pl_education_level=pl_1.fit_transform(education_level_array)
lr_2=LinearRegression()
lr_2.fit(pl_education_level,salary)

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,lr_2.predict(pl_1.fit_transform(education_level_array)),color='blue')
plt.show()

#predit
print(lr.predict(11))
print(lr.predict(6.6))

print(lr_2.predict(pl_1.fit_transform(11)))
print(lr_2.predict(pl_1.fit_transform(6.6)))
