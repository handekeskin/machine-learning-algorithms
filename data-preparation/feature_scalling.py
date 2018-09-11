import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer #eksik veriler için değişiklik yapmaya yardımcı oluyor
from sklearn.preprocessing import LabelEncoder#kategorik dataları sayısal hale getiriyor
from sklearn.preprocessing import OneHotEncoder#kategorik olarak sayısal hale gelen dataları kolonlara getirip flagleme işlemi yapıyor
from sklearn.cross_validation import train_test_split#test ve öğrenme datası ayırmak için kullanılıyor
from sklearn.preprocessing import StandardScaler#verilerin standartlaştırılması için kullanılıyor. Formulü (veri-ortalama değer)/st sapma

#csv'den data okuma
missing_values_data = pd.read_csv('data/missing_values.csv')

#eksik veri tamamlama
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)#eksik veriler için sütünların ortalamasını al koy demek.
yas = missing_values_data.iloc[:,1:4].values
print(yas)
imputer =imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

#kategorik data oluşturma
ulke = missing_values_data.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#numpy dizileri dataframe dönüşümü
sonuc1 = pd.DataFrame(data = ulke , index=range(22), columns=['fr','tr','us'])#array'ı dataframe'e çevirdi
print(sonuc1)

sonuc2= pd.DataFrame(data = yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=missing_values_data.iloc[:,-1:].values

sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22), columns=['cinsiyet'])

#data birleştirme
s1=pd.concat([sonuc1,sonuc2],axis=1)
print(s1)
s2=pd.concat([s1,sonuc3],axis=1)
print(s2)

#test - train datası oluşturma

x_train ,x_test , y_train, y_test = train_test_split(s1,sonuc3,test_size=0.33,random_state=0)

#standartlaştırma
st = StandardScaler()
X_train=st.fit_transform(x_train)
X_test=st.fit_transform(x_test)
