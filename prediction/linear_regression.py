import pandas as pd

data = pd.read_csv('../data/sales.csv')

#y=ax+b formülü kullanılarak doğru çizilir
#bagımlı değişken = katsayı * bağımsız değişken + sabit
#İlgili doğru mevcut noktaların doğruya olan uzaklarının minumum olacak şekilde hesaplanmasıyla bulunur.

#2.2 farklı yöntemler ile data seçme
aylar = data[['Aylar']]
satislar = data[['Satislar']]

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

'''
#verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train=st.fit_transform(x_train)
X_test=st.fit_transform(x_test)
Y_train=st.fit_transform(y_train)
Y_test=st.fit_transform(y_test)

#doğrusal regresyon
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,Y_train)
predict = lr.predict(X_test)
'''
#doğrusal regresyon
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
predict = lr.predict(x_test)