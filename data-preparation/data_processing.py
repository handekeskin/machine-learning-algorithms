import pandas as pd
import numpy as np

missing_values_data = pd.read_csv('data/missing_values.csv')

#1. method


#age_mean = np.mean(missing_values_data['yas'])

#missing_values_data['yas'].replace(np.nan, age_mean,inplace=True)

#print(missing_values_data)

#2.method

from sklearn.preprocessing import Imputer #eksik veriler için değişiklik yapmaya yardımcı oluyor

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)#eksik veriler için sütünların ortalamasını al koy demek.

yas = missing_values_data.iloc[:,3:4].values

print(yas)

imputer =imputer.fit(yas[:,:])

yas[:,:]=imputer.transform(yas[:,:])

print(yas)

#3. method


from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)#eksik veriler için sütünların ortalamasını al koy demek.

yas = missing_values_data.iloc[:,1:4].values

print(yas)

imputer =imputer.fit(yas[:,1:4])

yas[:,1:4]=imputer.transform(yas[:,1:4])

print(yas)



