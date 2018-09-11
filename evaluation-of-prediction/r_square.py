#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#R square yönteminde hesaplama yaparken 2 değer hesaplanır.
#hata kareleri toplamı(HKT)=topla((gerçek y değeri - hesapladığımız y değeri )^2)
#ortalama farkların toplamı(OFT) =topla ((gerçek y değeri - hesapladığımız y değerlerinin ortalaması)^2)
#R^2=1-(HKT/OFT)
#Ortalama farkların karesi bizim için en kötü olabilecek durumu gösteriyor.
#Mesela ben hiç bir tahmin yöntemi kullanmasamve elimdeki değerlerin ortalaması alıp yeni bir değer geldiğinde tahmin olarak ortama değeri versem
#tahmin metodları arasında çıkarabileceğim en kötü sonuç olması gerekir.
#tahmin yöntemlerinin en kötü base'i ortalama değeri vermektir. R-square değeri negatif çıkarsak ortalama değer vermektende kötü bir tahmin yapılmıştır. Bu hiç bir işe yaramaz
#eğer R square değerimiz 1 çıkarsa elde edebileceğimiz en iyi tahmin yöntemini bulmuşuzdur.
#1 e ne kadar yakınsa başarı değeri o kadar yüksektir.
#herhangi bir tahmin algoritmasının başarısını ölçmek için bir değer elde ediyoruz. Bu bize elimizdeki tahmin yöntemlerinin başarısını ölçmemizi sağlıyor.
#r square modelinin sorunlarından biri sisteme yeni değişkenler eklediğimizde bu değişken tahmin metodunu olumsuzda etkilese her zaman olumlu değer sağlar.
#olumlu etkisi olduğunda zaten R square değeri aratacaktır.
#olumsuz durumda örneğin bir multilinear regression modeli alalım. y=ax0+bx1+c denklemi üzerinden bir tahmin yöntemi oluştup R squre değeri bulalım.
#burada daha sonra modelimizi biraz daha iyileştirmek için y=ax0+bx1+cx2+d yeni bir değşken daha ekleyelim.
#bu değer olumsuz etki yapıyorsa tahmin metodumuza burda x2 değerinin katsayısı 0'a yakın bir değer olur.
#0a yakın olmasında da HKT değerini hesaplarken hata değerini olumsuz etkisi olmasına rağmen hesaplanılan y değerleri ile farkın küçülmesine sebep olur burda Rsquare değerini yükseltir.
#Yani modele yeni bir değişken eklediğimizde hiç bir zaman r square değeri bundan olumsuz etkilenmez.
#yeni değerin sisteme ne kadar olumlu katkısı olduğunu görmek için düzeltilmiş(Adjested R Square) değeri kullanılır.
# düzeltilmiş r square = 1 - (1-R^2)(n-1)/(n-p-1) --p değişken sayısı -- n eleman sayısı

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

#r2 score evaluation in linear regression
from sklearn.metrics import r2_score
print("evaluation of lineer regression:")
r2_score_evaluation_lr=r2_score(salary,lr.predict(education_level_array))
print(r2_score_evaluation_lr)

#polynomial-regression (2. degree)
from sklearn.preprocessing import PolynomialFeatures
pl_1=PolynomialFeatures(degree=2)
pl_education_level=pl_1.fit_transform(education_level_array)
lr_2=LinearRegression()
lr_2.fit(pl_education_level,salary)

#r2 score evaluation in polynomial-regression (2. degree)
from sklearn.metrics import r2_score
print("evaluation of polynomial-regression (2. degree):")
r2_score_evaluation_lr_2=r2_score(salary,lr_2.predict(pl_1.fit_transform(education_level_array)))
print(r2_score_evaluation_lr_2)

#polynomial-regression (4. degree)
pl_2=PolynomialFeatures(degree=4)
pl_education_level=pl_2.fit_transform(education_level_array)
lr_3=LinearRegression()
lr_3.fit(pl_education_level,salary)

#r2 score evaluation in polynomial-regression (4. degree)
from sklearn.metrics import r2_score
print("evaluation of polynomial-regression (4. degree):")
r2_score_evaluation_lr_3=r2_score(salary,lr_3.predict(pl_2.fit_transform(education_level_array)))
print(r2_score_evaluation_lr_3)

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
from sklearn.metrics import r2_score
print("evaluation of SVR:")
r2_score_evaluation_svr=r2_score(Y_scale,svr.predict(X_scale))
print(r2_score_evaluation_svr)

#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(education_level_array,salary_array)

#r2 score evaluation in Decision Tree
from sklearn.metrics import r2_score
print("evaluation of random forest:")
r2_score_evaluation_dt=r2_score(salary_array,r_dt.predict(education_level_array))
print(r2_score_evaluation_dt)

#random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(education_level_array,salary_array)

#r2 score evaluation in random-forest
from sklearn.metrics import r2_score
print("evaluation of random forest:")
r2_score_evaluation_rf=r2_score(salary_array,rf.predict(education_level_array))
print(r2_score_evaluation_rf)

