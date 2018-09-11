#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#SVM sınıflandırma problemleri için kullanılıyor. 2 yada daha fazla kümeyi vector ile birbirinden yarıyor.
#SVM 2 sınıfı en iyi ayırabilecek vector bulunmaya çalışıyor. Yani marjin aralığı en yüksek çizgiyi çekiyor
#SVR da ise maksimum noktayı içine alabilecek bir vector  çizmeyi ve bu çizdiği marjin aralığında maksimum nokta olmasını istiyor.
#yani bir marjin aralığına maksimum noktayı sığdıran vectoru bulmaya yarıyor.
#y=wx+b+epsilon y=wx+b y=wx+b-epsilon diye 3 doğru düşünürsek epsilon buarada marjin aralığı diye düşünülebilir.
#doğruyu epsilon kadar yukarı yada aşağıya kaydırdığımızda bizim marjin aralığımız belirleniyor ve bu araya max veri kümesi dahil edilmeye çalışılıyor.
#eğrilerde benzer bir mantık var eğri denklemine epsilon kadar ekleme ve çıkarma yapılıp marjin aralığı belirleniyor ve bu aralıkta max veri kümesi olması sağlanıyor.
#marjin aralığı küçük olan doğru bizim için daha anlamlı ve kıymetlidir.
#SVR da ki dejevantaj outlierlara karşı çok hassas olması polinom regresyonda en büyük farkı bu.
#bu svr modelini kullanırken dataları scale etmemiz gerekiyor yani ölçeklendirmemiz gerekiyor.
#en çok kullanılan kernel değeri rbf(radial bas function)

#2.1 veri yükleme(data load)
salary_data = pd.read_csv('../data/salary.csv')

#dataframe dilimleme(slice)
education_level=salary_data.iloc[:,1:2]
salary=salary_data.iloc[:,2:]

#numpy array dönüşümü
education_level_array=education_level.values
salary_array=salary.values

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

plt.scatter(X_scale,Y_scale,color='red')
plt.plot(X_scale,svr.predict(X_scale),color='blue')
plt.title('rbf')
plt.show()

#SVR model for linear
svr=SVR(kernel='linear')
svr.fit(X_scale,Y_scale)

plt.scatter(X_scale,Y_scale,color='red')
plt.plot(X_scale,svr.predict(X_scale),color='blue')
plt.title('linear')
plt.show()

#SVR model for polynomial
svr=SVR(kernel='poly')
svr.fit(X_scale,Y_scale)

plt.scatter(X_scale,Y_scale,color='red')
plt.plot(X_scale,svr.predict(X_scale),color='blue')
plt.title('polinomial')
plt.show()

#SVR model for sigmoid
svr=SVR(kernel= 'sigmoid')
svr.fit(X_scale,Y_scale)

plt.scatter(X_scale,Y_scale,color='red')
plt.plot(X_scale,svr.predict(X_scale),color='blue')
plt.title('sigmoid')
plt.show()

print(svr.predict(11))
print(svr.predict(6.6))



