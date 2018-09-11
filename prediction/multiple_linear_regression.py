#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# y = E+ a+ bx + cy + dz  şeklinde kaç değişken varsa o kadar katsayı bulunur. E hata payı
# burada değişkenler 1. dereceden bir formüldür yani çok boyutlu uzayda bir doğru ifade eder.
# çoklu regresyonda korelasyonu yüksek olan birden fazla kolonu aynı anda regresyona sokmak hataya sebep olur.
# Bunun nedeni çoklu regresyonda değişkenler bir kat sayı ile çarpılıp hesaplama yapılmaktadır. Birbirine benzer ifadeler yer alırsa
# bu o değişkenin baskın olmasına neden olur ve hataya sebep verir. Mesela cinsiyet kolonumuz varken bir de kadın_flg yada erkek_flg datası
# 3 kolonda aynı anlamdadır bunlardan birini almak yeterli olur. Bu değişkenlere dummy yada kukla değişken denir

#2.1 veri yükleme
data = pd.read_csv('../data/data.csv')

#encoder ordinal , nominal -> numeric (kategorik dataları numeric datalara çevirme)
from sklearn.preprocessing import LabelEncoder
ulke = data.iloc[:,0:1].values
le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

cinsiyet= data.iloc[:,-1:].values
le = LabelEncoder()
cinsiyet[:,0]=le.fit_transform(cinsiyet[:,0])
print(cinsiyet)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

ohe = OneHotEncoder(categorical_features='all')
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

#numpy dizileri dataframe dönüşümü
sonuc1 = pd.DataFrame(data = ulke , index=range(22), columns=['fr','tr','us'])#array'ı dataframe'e çevirdi
print(sonuc1)

yas = data.iloc[:,1:4].values

sonuc2= pd.DataFrame(data = yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc2)

sonuc3= pd.DataFrame(data=cinsiyet[:,0:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)

#data birleştirme
s1=pd.concat([sonuc1,sonuc2],axis=1)
print(s1)
s2=pd.concat([s1,sonuc3],axis=1)
print(s2)

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(s1,sonuc3,test_size=0.33,random_state=0)

#çoklu lineer regresyon modeli-cinsiyet için-
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_prediction =  mlr.predict(x_test)

print(mlr_prediction)
print(y_test)

#çoklu lineer regresyon modeli-boy için-

sag = s2.iloc[:,4:]
sol = s2.iloc[:,:3]
boy= s2.iloc[:,3:4]

s3=pd.concat([sag,sol],axis=1)
print(s3)

#test-train datası oluşturma
x_train_boy ,x_test_boy , y_train_boy, y_test_boy = train_test_split(s3,boy,test_size=0.33,random_state=0)

mlr.fit(x_train_boy,y_train_boy)
mlr_prediction_boy =  mlr.predict(x_test_boy)

print(mlr_prediction_boy)
print(y_test_boy)