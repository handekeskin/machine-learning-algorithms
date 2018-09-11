#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#birden fazla karar ağacı algoritmasının aynı veri seti için kullanılması ve
#problemin çözümünde bu algoritmaların hepsinin beraber kullanılarak hesaplama yapılmasıdır.
#buna çoğunluğun oyuda(majority vote) deniyor.
#normal decision tree de tüm data setini verip algoritmayı eğitiyorsun. Bu yüzden aynı veri setini bir kaç kez decision tree algoritmasınada versek aynı sonucu üretiyor.
#fakat random forest da data setinden rastgele datalar seçilerek küçük veriseti kümeleri ayrılıyor ve her ayrılan parça için decision tree algoritması üretiliyor.
#böylece elimizde birden fazla decision tree algoritması oluyor ve yeni datayı tahmin ederken tüm decision treelerden gelen sonuçlar alınıyor
#ve çıkan sonuçların ortalaması alınarak bizim değelerimiz hesaplanıyor. Bu tahmin kısmı için
#Sınıflandırmada en çok tekrar eden değer doğru kabul ediliyor.
#karar ağaçlarında veri artınca başarı düşüyor. bu yüzden büyük datalarda random forest data etkili
#çok data olunca overfiting yani datayı ezberleme problemi karşımıza çıkıyor.
# Data çok olunca data çok data dallanıyor ve hesaplama süresi uzuyor.
#random forestla birden fazla bakış açısı sağlanabiliyor.
#ensable kollektif öğrenme metodu
#n_estimators kaç tane decision tree çizileceği bilgisi
#decision tree bildiği verileri tahmin etmede çok iyi ama bilmediği verilerde bildiği değelerden birini getirme eğiliminde ama random forest yeni değer getirme kapasitesine sahip

#2.1 veri yükleme(data load)
salary_data = pd.read_csv('../data/salary.csv')

#dataframe dilimleme(slice)
education_level=salary_data.iloc[:,1:2]
salary=salary_data.iloc[:,2:]

#numpy array dönüşümü
education_level_array=education_level.values
salary_array=salary.values

#random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(education_level_array,salary_array)
z=education_level_array+0.5
k=education_level_array-0.4

plt.scatter(education_level_array,salary_array,color='red')
plt.plot(education_level_array,rf.predict(education_level_array),color='blue')
plt.plot(education_level_array,rf.predict(z),color='green')
plt.plot(education_level_array,rf.predict(k),color='yellow')
plt.show()

print(rf.predict(11))
print(rf.predict(6.6))