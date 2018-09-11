#sayısal verilerin tahmini prediction sayısal olmayan veriler için classification
#logistic regresyon formülü
#sigma(t)= 1/(1+e^(-t))
#t=b0+b1x  t=A+BX  NOT:B ne kadar büyürse logistic regresyon eğrisi s kısmı o kadar dik olur. küçülüncede s doğruyu yaklaşıyor
#p(x)=1/(1+e^-(b0+b1x))
#birden fazla değişken varsa t=b0+b1x1+b2x2+b3x3+...+bnxn

#karmaşıklık matrisi
#            C1                 C2
#C1     true positive      false negative
#C2     false positive     true negative

#C1 im ve tahlillerde de C1 bulundum yani kanserdim kanser çıktım : true positive (true doğru sonuç tahmini positive kanser olmam)
#C1 im ve tahlillerde de C2 bulundum yani kanserdim kanser değilim çıktım : false positive (false yanlış sonuç tahmini positive kanser olmam)
#C2 yim ve tahlillerde de C1 bulundum yani kanser değildim ama kanser çıktım : false negative (false yanlış sonuç tahmini nagative kanser olmamam)
#C2 yim ve tahlillerde de C2 bulundum yani kanser değildim ve kanser değilim çıktım : true negative (true doğru sonuç tahmini nagative kanser olmamam)

#accuracy M : acc(M) model için yüzde kaç doğru sınıflandırma olduğu
#error rate (misclassification rate)= 1-acc(M)
#sensivity (recall) = true positive /(true positive+false negative) #true positive regression rate#
#specifity = true negative /(true negative+false positive ) #true negative regression rate#
#prediction = true positive /(true positive + true positive )
#negative predictive value  = true nagative /(true negative +false negative)
#accurancy = sensitivity*pos/(pos+neg) + specifity*neg/(neg+pos)
#örneğin C1 kanser olmak ve C2 de tam tersi kanser olmamak olsun
#sütun olarak yer alanlar benim gerçek değerlerim satır olarak yer alanlar tahmin değerlerim
#sensivity kaç kişinin kanser olduğunu doğru tahmin ettim.
#specifity kanser olmayanların kaçının doğru tahmin edildiği
#prediction kanser olanların ne kadar doğru tahmin olduğu

#matriste köşegende yer alanlar başarılı tahminlerdir
#diyagon dışındakilerse hataları verir
#doğru bildiğim örnekler / tüm küme bana doğruluk - başarı yüzdemi verir aşağıda formulü var
#(true positive+ true negative) / (true positive+ true negative+ false positive+ false negative )=acc(M)
#yanlış bildiğim örnekler / tüm küme bana hata - başarısızlık yüzdemi verir aşağıda formulü var
#(false positive+ false negative) / (true positive+ true negative+ false positive+ false negative )=1-acc(M)


#1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.veri ön işleme

#2.1 veri yükleme
data = pd.read_csv('../data/data.csv')

#bağımlı ve bağımsız değişkenleri ayırma
x = data.iloc[:,1:4].values
y=data.iloc[:,-1:].values

#verilerin test ve train olarak bölünmesi
from sklearn.cross_validation import train_test_split
x_train ,x_test , y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train=st.fit_transform(x_train)
X_test=st.transform(x_test)

#Logistic regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)
print(y_pred)
print(y_test)

#The confusion matrix
from sklearn.metrics import confusion_matrix
con_met=confusion_matrix(y_test,y_pred)
print(con_met)
