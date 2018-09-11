import pandas as pd
from sklearn.preprocessing import LabelEncoder#kategorik dataları sayısal hale getiriyor
from sklearn.preprocessing import OneHotEncoder#kategorik olarak sayısal hale gelen dataları kolonlara getirip flagleme işlemi yapıyor

data = pd.read_csv('data/data.csv')

print(data)

ulke = data.iloc[:,0:1].values

print(ulke)

le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)