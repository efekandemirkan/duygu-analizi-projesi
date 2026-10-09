# duygu-analizi-projesi
# FER-2013 Tabanlı Yüz İfadelerinden Duygu Tanıma Sistemi

**İngilizce Proje Adı:** Facial Emotion Recognition Using the FER-2013 Dataset

## 1. Proje Hakkında

Bu proje, bilgisayarlı görü (Computer Vision), görüntü işleme (Image Processing) ve derin öğrenme (Deep Learning) tekniklerinden yararlanarak insan yüz ifadelerinden duygu sınıflandırması gerçekleştiren bir sistem geliştirmeyi amaçlamaktadır. Projede, yüz ifadelerinin analiz edilmesi ve bu ifadelerin önceden tanımlanmış duygu kategorilerinden birine atanması için FER-2013 (Facial Expression Recognition 2013) veri seti kullanılacaktır.

Sistem, veri setinde bulunan etiketlenmiş yüz görüntülerinden öğrenerek farklı yüz ifadelerini birbirinden ayırt edebilen bir derin öğrenme modeli oluşturacaktır. Modelin eğitilmesinin ardından daha önce karşılaşmadığı yüz görüntüleri üzerinde tahmin yapması ve elde edilen sonuçların çeşitli performans ölçütleriyle değerlendirilmesi hedeflenmektedir.

Projenin temel amacı, ham görüntülerden anlamlı özellikler çıkarabilen ve bu özelliklerden hareketle yüz ifadelerini sınıflandırabilen bir yapay zekâ sistemi geliştirmek; görüntü ön işleme, veri hazırlama, model eğitimi ve performans değerlendirme aşamalarını bütünleşik bir iş akışı içerisinde gerçekleştirmektir.

Proje geliştirme süreci boyunca doğruluk oranının yanı sıra sınıflar arasındaki performans farklılıkları ve modelin karşılaştığı sınıflandırma hataları da incelenecektir.

## 2. Projenin Amacı ve Hedefleri

Projenin genel amacı, FER-2013 veri seti kullanılarak yüz ifadelerini otomatik olarak sınıflandırabilen bir derin öğrenme modeli geliştirmektir.

Bu amaç doğrultusunda aşağıdaki çalışmaların gerçekleştirilmesi planlanmaktadır:

- FER-2013 veri setinin incelenmesi ve görüntülerin sınıf etiketleriyle birlikte analiz edilmesi.
- Eğitim ve test verilerinin uygun şekilde okunması ve hazırlanması.
- Görüntülerin model eğitimine uygun biçime dönüştürülmesi.
- Görüntüler üzerinde gerekli ön işleme ve normalizasyon işlemlerinin uygulanması.
- Yüz ifadelerinden ayırt edici özellikleri öğrenebilen bir Evrişimsel Sinir Ağı (Convolutional Neural Network — CNN) modelinin geliştirilmesi.
- Modelin eğitim verileri kullanılarak eğitilmesi ve doğrulama verileriyle izlenmesi.
- Model performansının test verileri üzerinde ölçülmesi.
- Yanlış sınıflandırmaların analiz edilmesi ve modelin iyileştirilmesine yönelik çalışmalar yapılması.
- Geliştirilen modelin yeni yüz görüntüleri üzerinde tahmin yapabilecek şekilde hazırlanması.

## 3. Kullanılacak Veri Seti: FER-2013

Projede, yüz ifadelerinin otomatik olarak sınıflandırılması amacıyla oluşturulmuş FER-2013 veri setinden yararlanılacaktır.

FER-2013, yüz ifadelerini yedi farklı kategori altında sınıflandırmak için kullanılan, gri tonlamalı yüz görüntülerinden oluşan bir veri setidir. Veri setindeki standart görüntüler 48 × 48 piksel çözünürlüğündedir. Görüntüler, farklı yüz ifadelerinin öğrenilmesi ve bu ifadelerin birbirinden ayırt edilmesi amacıyla etiketlenmiştir.

FER-2013 veri seti, yüz ifadelerini tanıma problemini gerçekçi koşullara daha yakın hâle getiren çeşitli görsel farklılıklar içerir. Yüzlerin görünümü, ifade yoğunluğu, poz ve görüntü kalitesi gibi etkenler sınıflandırma sürecini zorlaştırabilir. Ayrıca duygu sınıflarındaki örnek sayıları eşit olmadığından, modelin yalnızca genel doğruluk oranı üzerinden değerlendirilmesi yeterli olmayabilir.

Bu nedenle proje kapsamında veri setinin sınıf dağılımı incelenecek, görüntülerin eğitim öncesindeki durumu kontrol edilecek ve model başarısı her bir duygu sınıfı açısından ayrıca değerlendirilecektir.

## 4. Duygu Sınıfları

FER-2013 veri setinde yedi farklı yüz ifadesi sınıfı bulunmaktadır. Projede modelin aşağıdaki kategoriler arasında sınıflandırma yapması hedeflenmektedir.

| Sınıf | İngilizce Etiket | Açıklama |
|---|---|---|
| Öfkeli | `angry` | Öfke ifadesi içeren yüz görüntüleri |
| Tiksinmiş | `disgust` | Tiksinme ifadesi içeren yüz görüntüleri |
| Korkmuş | `fear` | Korku ifadesi içeren yüz görüntüleri |
| Mutlu | `happy` | Mutluluk veya gülümseme ifadesi içeren yüz görüntüleri |
| Nötr | `neutral` | Belirgin bir duygu ifadesinin bulunmadığı yüz görüntüleri |
| Üzgün | `sad` | Üzüntü ifadesi içeren yüz görüntüleri |
| Şaşkın | `surprise` | Şaşkınlık ifadesi içeren yüz görüntüleri |

Sınıf etiketleri, görüntülerin ait olduğu kategoriyi belirtmek için kullanılacaktır. Model, eğitim sırasında görüntüler ile bu etiketler arasındaki ilişkileri öğrenerek yeni görüntüler için sınıf tahminleri üretecektir.

Burada amaç, kişinin gerçek iç dünyasındaki duyguyu kesin olarak belirlemek değil, görüntüde gözlemlenen yüz ifadesini veri setindeki kategorilerden biriyle eşleştirmektir.

## 5. Görüntü Ön İşleme Süreci

Derin öğrenme modelinden verimli sonuçlar elde edilebilmesi için görüntülerin belirli bir düzende hazırlanması gerekmektedir. Bu nedenle model eğitiminden önce bir görüntü ön işleme aşaması gerçekleştirilecektir.

Planlanan ön işleme adımları şunlardır:

### 5.1. Görüntülerin okunması

Eğitim ve test klasörlerindeki görüntüler Python aracılığıyla okunacaktır. Her görüntünün hangi duygu kategorisine ait olduğu, bulunduğu klasör üzerinden belirlenecektir.

### 5.2. Veri kontrolü

Görüntülerin okunabilirliği, boyutları ve etiketlerinin doğruluğu kontrol edilecektir. Bozuk, eksik veya beklenen biçime uymayan dosyaların model eğitimini olumsuz etkilemesi önlenecektir.

### 5.3. Görüntü boyutlarının kontrolü

FER-2013 veri setinin standart görüntüleri 48 × 48 piksel boyutundadır. Görüntülerin boyutları kontrol edilecek ve gerekli görülürse modelin beklediği giriş boyutuna dönüştürülecektir.

### 5.4. Gri tonlama

Görüntülerin gri tonlamalı biçimde kullanılması planlanmaktadır. Görüntüler zaten gri tonlamalıysa gereksiz dönüşümlerden kaçınılacak; farklı bir formatta bulunan görüntüler uygun biçime getirilecektir.

### 5.5. Normalizasyon

Görüntülerin piksel değerleri, modelin daha istikrarlı öğrenmesine yardımcı olmak amacıyla uygun bir sayısal aralığa dönüştürülecektir. Örneğin, 0–255 aralığındaki piksel değerleri 0–1 aralığına ölçeklendirilebilecektir.

### 5.6. Etiketlerin hazırlanması

Her görüntünün duygu sınıfı, modelin anlayabileceği etiket biçimine dönüştürülecektir. Eğitim aşamasında görüntü verileri ile doğru sınıf etiketleri birlikte kullanılacaktır.

## 6. Kullanılacak Yöntem: Evrişimsel Sinir Ağı (CNN)

Projede ilk model yaklaşımı olarak Evrişimsel Sinir Ağı (Convolutional Neural Network — CNN) kullanılması planlanmaktadır.

CNN mimarileri, görüntülerdeki uzamsal örüntüleri öğrenmek amacıyla yaygın olarak kullanılan derin öğrenme modelleridir. Yüz ifadelerinin sınıflandırılması probleminde modelin yüz bölgesindeki anlamlı görsel özellikleri öğrenmesine yardımcı olabilir.

Modelin temel çalışma mantığının aşağıdaki bileşenlerden oluşması öngörülmektedir:

- **Giriş katmanı:** Ön işleme aşamasından geçen yüz görüntülerini modele aktarır.
- **Evrişim katmanları:** Görüntülerdeki çizgi, kenar, doku ve daha karmaşık görsel örüntüleri öğrenir.
- **Aktivasyon fonksiyonları:** Ağın doğrusal olmayan ilişkileri öğrenmesini sağlar.
- **Havuzlama katmanları:** Görsel özellik haritalarının boyutunu azaltarak hesaplama yükünü sınırlandırır.
- **Düzleştirme ve tam bağlantılı katmanlar:** Öğrenilen özelliklerin sınıflandırma için kullanılmasını sağlar.
- **Çıkış katmanı:** Yedi duygu sınıfına yönelik tahmin değerlerini üretir.

Çıkış katmanı, her görüntü için yedi sınıfa ait tahmin değerleri oluşturacak şekilde tasarlanacaktır. Uygun bir sınıflandırma yaklaşımıyla bu değerlerden en yüksek olanı modelin tahmin ettiği sınıf olarak belirlenecektir.

Model mimarisi, katman sayısı ve diğer eğitim parametreleri deneysel sonuçlara göre belirlenecek; gerekirse farklı yapılandırmalar karşılaştırılacaktır.

## 7. Modelin Eğitim Süreci

Ön işleme aşaması tamamlandıktan sonra eğitim görüntüleri kullanılarak CNN modelinin eğitilmesi planlanmaktadır.

Eğitim sürecinde model, görüntülerden öğrendiği özellikleri doğru duygu sınıflarıyla eşleştirmeye çalışacaktır. Tahminler ile gerçek etiketler arasındaki fark, uygun bir kayıp fonksiyonu yardımıyla ölçülecek; optimizasyon algoritması aracılığıyla modelin parametreleri güncellenecektir.

Eğitim aşamasında aşağıdaki çalışmalar yürütülecektir:

1. Eğitim verilerinin modele aktarılması.
2. Model mimarisinin oluşturulması.
3. Uygun kayıp fonksiyonu ve optimizasyon yönteminin belirlenmesi.
4. Modelin belirlenen epoch sayısı boyunca eğitilmesi.
5. Eğitim ve doğrulama sonuçlarının izlenmesi.
6. Aşırı öğrenme (overfitting) belirtilerinin kontrol edilmesi.
7. Gerekli görülürse öğrenme oranı, katman yapısı ve diğer hiperparametrelerin iyileştirilmesi.
8. En uygun modelin kaydedilmesi ve daha sonra kullanılabilecek hâle getirilmesi.

Modelin doğrulama performansını izlemek için eğitim verilerinin bir bölümü doğrulama amacıyla ayrılacaktır. Böylece model seçimi yalnızca eğitim başarısına dayandırılmayacaktır.

## 8. Model Performansının Değerlendirilmesi

Eğitilen modelin başarısı, eğitim aşamasında kullanılmayan test görüntüleri üzerinden değerlendirilecektir. Bu değerlendirme, modelin yeni görüntülerdeki sınıflandırma performansı hakkında daha güvenilir bir ölçüm elde edilmesini sağlayacaktır.

Performans değerlendirmesinde aşağıdaki ölçütlerin kullanılması planlanmaktadır:

**Accuracy (Doğruluk):** Tüm test görüntüleri içerisinde doğru sınıflandırılan görüntülerin oranını gösterir.

**Precision (Kesinlik):** Modelin belirli bir sınıf için yaptığı tahminlerin ne kadarının doğru olduğunu ölçer.

**Recall (Duyarlılık):** Gerçekte belirli bir sınıfa ait olan görüntülerin ne kadarının model tarafından doğru tespit edildiğini gösterir.

**F1-Score:** Precision ve Recall değerlerini birlikte değerlendiren bir ölçüttür. Özellikle sınıflar arasındaki örnek sayılarının farklı olduğu durumlarda önemlidir.

**Confusion Matrix (Karışıklık Matrisi):** Gerçek sınıflarla modelin tahmin ettiği sınıfları karşılaştırarak hangi duygu kategorilerinin birbiriyle karıştırıldığını gösterir.

Bu ölçütler, modelin yalnızca genel başarısını değil, her bir duygu sınıfındaki güçlü ve zayıf yönlerini de incelemek için kullanılacaktır. Özellikle örnek sayısı düşük olan kategorilerin performansı ayrıca değerlendirilecektir.

Model henüz eğitilip test edilmediği için bu aşamada herhangi bir başarı oranı veya kesin performans sonucu belirtilmemektedir. Ölçülen sonuçlar elde edildikten sonra proje dokümantasyonuna eklenecektir.

## 9. Kullanılması Planlanan Teknolojiler

Projenin geliştirilmesinde aşağıdaki araç ve teknolojilerden yararlanılması planlanmaktadır:

| Teknoloji | Kullanım Amacı |
|---|---|
| Python | Projenin ana programlama dili |
| NumPy | Sayısal işlemler ve dizi işlemleri |
| OpenCV | Görüntü okuma ve görüntü işleme işlemleri |
| TensorFlow / Keras | Derin öğrenme modelinin geliştirilmesi ve eğitimi |
| Matplotlib | Eğitim sonuçlarının ve verilerin görselleştirilmesi |
| scikit-learn | Performans ölçütlerinin hesaplanması ve karışıklık matrisinin oluşturulması |
| Git ve GitHub | Sürüm kontrolü, kod yönetimi ve proje dokümantasyonu |
| Python sanal ortamı (`venv`) | Proje bağımlılıklarının yalıtılması |

Kullanılacak kütüphaneler ve sürümleri, model geliştirme süreci netleştikçe belirlenerek bağımlılık dosyasında kayıt altına alınacaktır.

## 10. Geliştirme Aşamaları

Projenin sistematik bir biçimde ilerletilmesi için geliştirme sürecinin aşağıdaki aşamalara ayrılması planlanmaktadır:

**Aşama 1 — Veri setinin incelenmesi:** FER-2013 veri setinin klasör yapısının, görüntü özelliklerinin ve sınıf dağılımlarının analiz edilmesi.

**Aşama 2 — Ön işleme altyapısı:** Görüntülerin okunması, kontrol edilmesi, etiketlenmesi ve model girişine uygun hâle getirilmesi.

**Aşama 3 — Model geliştirme:** İlk CNN mimarisinin oluşturulması ve eğitim sürecinin yapılandırılması.

**Aşama 4 — Model eğitimi:** Eğitim verileri kullanılarak modelin eğitilmesi ve doğrulama sonuçlarının takip edilmesi.

**Aşama 5 — Performans analizi:** Test verileri üzerinde ölçümlerin gerçekleştirilmesi, karışıklık matrisinin oluşturulması ve hatalı tahminlerin incelenmesi.

**Aşama 6 — İyileştirme:** Gerekli durumlarda veri artırma, sınıf ağırlıkları ve model parametreleri gibi unsurların değerlendirilmesi.

**Aşama 7 — Tahmin sistemi:** Eğitilmiş modelin yeni yüz görüntülerini analiz ederek bir duygu sınıfı tahmini üretebilmesi için gerekli kodun hazırlanması.

**Aşama 8 — Dokümantasyon:** Kullanılan yöntemlerin, deneylerin, değerlendirme sonuçlarının ve projenin sınırlılıklarının belgelenmesi.

Gerekli temel aşamalar tamamlandıktan sonra, proje kapsamına uygun olması durumunda canlı kamera görüntülerinden tahmin gerçekleştirebilen bir uygulama geliştirilmesi de değerlendirilecektir. Bu özellik, mevcut aşamada tamamlanmış bir işlev olarak değil, olası bir genişletme olarak ele alınmaktadır.

## 11. Beklenen Çıktılar

Projenin tamamlanmasıyla birlikte aşağıdaki çıktıların elde edilmesi hedeflenmektedir:

- FER-2013 veri setini okuyabilen ve eğitim için hazırlayabilen bir veri işleme altyapısı.
- Yedi duygu kategorisini sınıflandırmak üzere eğitilmiş bir CNN modeli.
- Eğitim ve doğrulama süreçlerini takip etmeye yönelik sonuçlar ve grafikler.
- Test verileri üzerinde hesaplanmış performans ölçütleri.
- Sınıflandırma hatalarını analiz etmeye yarayan bir karışıklık matrisi.
- Yeni yüz görüntülerinden duygu sınıfı tahmini üretebilen bir Python uygulaması.
- Projenin yöntemlerini ve deneysel sonuçlarını açıklayan teknik dokümantasyon.

Bu çıktılar, proje ilerledikçe oluşturulacak ve elde edilen gerçek sonuçlara göre güncellenecektir.

## 12. Projenin Sınırlılıkları ve Dikkat Edilecek Noktalar

Yüz ifadelerinden duygu sınıflandırması, görüntü kalitesi, aydınlatma, yüzün açısı, ifade yoğunluğu ve kişiler arasındaki görsel farklılıklar gibi etkenlerden etkilenebilir. Ayrıca FER-2013 veri setindeki sınıf dağılımlarının eşit olmaması, modelin bazı kategorilerde diğerlerine kıyasla daha başarılı veya başarısız olmasına neden olabilir.

Bu nedenle proje kapsamında genel doğruluk oranının tek başına yeterli bir performans göstergesi olarak kabul edilmemesi; her sınıf için Precision, Recall ve F1-Score değerlerinin de değerlendirilmesi önemlidir.

Bunun yanında, bir yüz ifadesi kişinin gerçek duygusal durumunu kesin olarak göstermeyebilir. Sistem yalnızca görüntüdeki gözlemlenebilir ifadeleri veri setinde tanımlanan kategoriler doğrultusunda sınıflandırmak üzere tasarlanacaktır. Model sonuçları, kişinin psikolojik durumu veya niyetleri hakkında kesin bir çıkarım olarak değerlendirilmemelidir.

Modelin farklı kişi grupları ve farklı görüntü koşulları üzerindeki genellenebilirliği de veri setinin kapsamı ve elde edilen sonuçlar doğrultusunda yorumlanmalıdır.

## 13. Projenin Mevcut Durumu

Proje şu anda geliştirme aşamasındadır. Veri seti, `train` ve `test` klasörleri altında yedi duygu sınıfına ayrılmış görüntülerden oluşmaktadır. `preprocess.py` dosyası, veri ön işleme altyapısının geliştirilmesi için proje içerisinde yer almaktadır.

Bir sonraki aşamalarda veri hazırlama sürecinin tamamlanması, CNN modelinin geliştirilmesi, modelin eğitilmesi ve test sonuçlarının analiz edilmesi planlanmaktadır.

Modelin başarı oranı, eğitim parametreleri ve deneysel bulgular henüz elde edilmediğinden bu bilgiler ölçümler tamamlandıktan sonra README dosyasına eklenecektir.

## 14. Katkı ve Geliştirme

Proje, Python ile görüntü işleme ve derin öğrenme uygulamalarını incelemek ve yüz ifadelerinden duygu sınıflandırması gerçekleştiren bir sistem geliştirmek amacıyla yürütülmektedir.

Geliştirme sürecinde veri işleme, model mimarisi, eğitim parametreleri ve değerlendirme yöntemleri sistematik olarak ele alınacaktır. Yapılan değişikliklerin ve deneylerin belgelenmesiyle projenin yeniden üretilebilirliğinin ve teknik olarak anlaşılabilirliğinin artırılması hedeflenmektedir.

**Proje Adı:** FER-2013 Tabanlı Yüz İfadelerinden Duygu Tanıma Sistemi  
**Ana Teknoloji:** Python  
**Yöntem:** Görüntü İşleme ve Derin Öğrenme  
**Planlanan Model Yaklaşımı:** Evrişimsel Sinir Ağı (CNN)  
**Veri Seti:** FER-2013  
**Sınıflandırma Türü:** Yedi sınıflı yüz ifadesi sınıflandırması  
**Durum:** Geliştirme aşamasında
