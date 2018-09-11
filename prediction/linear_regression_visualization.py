import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/sales.csv')

#2.2 data seçme
aylar = data[['Aylar']]
satislar = data[['Satislar']]

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)



#doğrusal regresyon
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
predict = lr.predict(x_test)

#print(predict)

#dataları sıralama
x_train = x_train.sort_index()
y_train = y_train.sort_index()

#grafik çizme

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title('aylara göre satış')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
plt.show()

