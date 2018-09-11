#bunu alanlar bunuda aldı. Bunu izleyenler bunuda izledi gibi çıkarım yapan kurallar
#sürekli tekrar eden eylemi yakalamak amacımız
#eğer bu ilişkileri bulabilirsek satışları kendi tarafımıza çekebiliriz.
#aradaki ilişkinin nedenini bulmadan ilişkiyi bulmak daha öenmli oluyor.
#örneğin köpek balığının insanlara saldırmasıyla dondurma satışları arasında bir ilişki var. Bunun nedeni hava sıcak olduğunda
#köpek balıkları insanlara saldırıyor ve hava sıcakken dondurma satışı artıyor. Buna göre hava sıcak olması koşulu öenmli olmadan
#dondurma satama ve köpek balığı saldırısı arasında ilişki bulmak yeterli olabilir
#her müşterinin yaptığı eylemleri izleyip onunla benzer eylemleri yapan insanlara daha önce diğer insanların yaptığı eylemler önerilebilir.
#support - destek
#support(a) = a varlığını içeren eylemler / tüm eylemler
#örneğin 100 kişiye a filimini izledin mi diye sorduk 10 kişi izlemiş 90 kişi ezlememiş
#support(a filminin izlenmesi) = 10 / 100 olur
#confidence - güven
#confidence(a->b)= a ve b varlığını içeren eylemler / a varlığını içeren eylemler
#örneğin daha önce fa filmini izlemiş 10 kişiden b filmini izlemiş 2 kişi var izlememiş 8 kişi
#confidence(a filmini izlemiş->b filmini izlemiş) = 2 / 10 olur
#tek bir üründen bahsettiğimizde support önemli ama bir ilişki arıyorsak confidence 'a bakılmalı
#lift
#lift(a->b)=confidence(a->b)/support(b)
#buradaki amaç a eyleminin b eylemi üzerindeki etkisini bulmak
#örneğin ilk 100 kişiye sorduğumuzda 10 kişi a filmini izlemiş.
#Bu yüz kişiden 12 kişide b filmini izlemiş olsun. Yukardan bildiğimiz gibi a ve b filmini izleyen 2 kişi vardı.
#lift(a filmini izlemiş->b filmini izlemiş) = (2/10)/(12/100)
#yani tüm grup içinde a grubuna odaklanmamız b ürününü satın alma olasılığını artırıyor mu
#bu örneğimizde genel kümeye baktığımızda b filminin izlenmesi %12 - a filmini izleyenlerde b filmini izleme %20
#yani a filmini izleyenlere konsantre olarak b filminin izlenmesi artırılabilir.
#lift değeri bu artışın n ekadar olacağını hesaplıyor. lift değeri 1 in altında çıkarsa a filminin izlenmesi b filmini izlenmesi olumsuz etkiliyor denebilir
#eğer lift değer 1'in üstünde ise a filminin izlenmesi b filminin izlenmesini olumlu etkiliyor demektir.

#apriori
#örneğin elimizde 4 ürün var önce bu ürünlerin support değerlerine bakıyoruz.
#bir support değeri belirleyip onun altında kalan ürünleri eleyebiliriz. Mesela 4 üründen birinin support değeri %3 olsun. Biz
#diğer ürünler ile ilişkisini incelesek bile bu ürünün toplumdaki değeri %3'ü geçmeyecek. o zaman bunu eleyip diğer ürünlerin birbiri ile ilişkilerine bakılabilir.
#sonrasında elimizde 3 ürün kaldı bu ürünleri birbiri arasındaki ikili ilişkiye bakıyoruz ve yine belirlediğimiz değerden düşük olanları eleyip
#geri kalan ilişkilerin nasıl olduğunu inceliyorz ve böylece apriori algoritması çalışıyor. Ekran görüntü ekledim bakılabilir.
#appriori kısaca frekans sayar belli değerden düşük kaldığında bu değerleri eler.
#birini ile yapılan eylemleri ele alıp bunların frekanslarıyla birleştirme yapılabilir. Bununda ilgili de apriori algoritmesı farklı örnek olarak ekran görüntü var.
#eclat algoritması küçük verilerde işe yarar fakat büyük verilerde sorun yaşar. Apriori büyük yerilerde daha iyi

#nerede kullanılıyor
#kampanya --temel kullanım amacı bu
#complex event processing -- örneğin bu havayoluyla uçan adam şu firmadan araba kiralar - bu cafede yemek yer gibi
#davranış tahmini--daha önceki müşterilerin eylemlerinden sonraki müşterinin eylemi tahmin edilebilir
#yönlendirilmiş arm-- a ürününü almakla b ürürünü almak arasında ilişki ile ve b ürününü almakla a ürününü almak arasındaki ilişki (apriori algoritmasında yön yoktur.)
#                    --örneğin a ürünü ilk yapan kişi b işini yaparken b işini ilk yapana kişi a işini hiç yapmıyor olabilr.
#zaman serisi analizi--mesela borsanın çıkış ve iniş zamanlarında politika dış siyaset frekanlarıyla bağlamanız.
#                   -- falanca olay oldu sonrasında filanca olay oldu sonucunda borsa düştü çıkarımı yapmak örneğin

#eclat (equivalance class transformation)
#aprioriden daha hızlı çalışır
#apriori algoritmasında ilk her ürünün frekansı bulunur belirlenen değerden az geçen ürünler elenir
#sonrasında kalanlar arasında ikili ilişkili frekanslara bakılır tekrar frekansın altında kalanlar elenir sonra kalanlar üçlü incelenir gibi bir döngü vardır
#eclat de ise her ürünün hangi transactionlarda geçtiğine bakılıyor.
#4 ürün olduğunda a-b-c-d de apprioride önce tek tek a, b,c,d ilişkisi çıkıyor sonra ab,ac,ad,.. ilişkileri çıkıyor sonra abc işilkisi çıkıyor
#fakat eclat de abcnin çıkması için d deki hiç bir şeyin hesaplanması gerekmiyor.
#breath first search(apriori) her seviye öğrendikten sonra diğerine geçiyor
#depth first seard (eclat) isteği seviyedeki öğrenmeyi hızlıca yapabiliyor.
#eclat ile ilgili ekran görüntünüsü ekledim.
#her ürünün hangi transactionlarda geçtiğini buluyoruz ürünler kolon transactionlar satır oluyor ve her yeni ilişkiyi yine kolon olarak alıp hangi transactionda geçtiğine bakıyoruz.
#aprioride ürünlerin kaç defa geçtiğini sayıyoruz. eclat de hangi transactionlarda geçtiğine bakıyoruz.
#apriori eclatın yaptığı işleri yapabiliyor ve büyük kümelerde daha iyi çalışıyor.