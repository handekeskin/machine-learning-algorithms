#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#burada datayı karar ağacı kullanarak belli bölgelere ayırıyoruz. Bu ayırma işlemi entropy'ye göre yapılıyor.
#Ayırma işlemi tamamlandıktan sonra ilgili karar ağacı kısmına giren verilerin ortalaması ile o bölgeye denk gelen ortalama değer bulunur.
#tahmin işlemi yapacağımız zaman bu karar ağacında bizim datamızın nerede kaldığına bakılarak hangi ortalama değere denk geldiği hesaplanır.
#burada ölçekleme yapmamız gerekmiyor. Ölçekleme yapmasakda kullanabiliriz.

#2.1 veri yükleme(data load)
salary_data = pd.read_csv('../data/salary.csv')

#dataframe dilimleme(slice)
education_level=salary_data.iloc[:,1:2]
salary=salary_data.iloc[:,2:]

#numpy array dönüşümü
education_level_array=education_level.values
salary_array=salary.values

#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(education_level_array,salary_array)
z=education_level_array+0.5
k=education_level_array-0.4

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,r_dt.predict(education_level_array),color='red')
plt.plot(education_level_array,r_dt.predict(z),color='green')
plt.plot(education_level_array,r_dt.predict(k),color='yellow')
plt.show()

print(r_dt.predict(11))
print(r_dt.predict(6.6))