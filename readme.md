# 🛰️ Yapay Zeka Destekli Canlı Orman Yangını Erken Uyarı ve Karar Destek Platformu

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/NASA-FIRMS%20API-red?style=for-the-badge&logo=nasa&logoColor=white" alt="NASA">
</p>

---

### 🌐 Canlı Uygulama Linki
Projenin interaktif web arayüzüne dünya çapında erişmek ve canlı simülasyonu incelemek için aşağıdaki bağlantıyı kullanabilirsiniz:  
🚀 **[Hugging Face Spaces - Canlı Takip Paneli](https://huggingface.co/spaces/sancos/Edge_Forest_Fire_Turkiye)**

---

## 📌 Proje Genel Bakışı
Bu çalışma; küresel iklim kriziyle birlikte artan orman yangını tehditlerine karşı, **Derin Öğrenme (Computer Vision)** ve **Coğrafi Bilgi Sistemlerini (GIS)** tek bir potada eriten hibrit bir erken uyarı prototipidir. 

Sistem iki ana omurgadan oluşur:
1. **Edge AI Sınıflandırma Motoru:** Gözlem dronları ve uydu donanımları (Raspberry Pi, Jetson Nano, Google Coral) için optimize edilmiş, **15.59 MB** boyutunda, **%97.40** doğruluk oranına sahip ultra hafif bir `EfficientNet-B0` derin öğrenme modeli.
2. **Coğrafi Takip Paneli (GIS):** NASA FIRMS uydu ağından gelen termal anomalileri anlık olarak süzüp, Kelvin-Santigrat dönüşümlerini mühendislik formülleriyle tamamlayarak interaktif haritaya işleyen canlı yönetim arayüzü.

---

## 🛠️ Sistem Mimarisi ve Veri Hattı (Pipeline)

### 1. Veri Mühendisliği ve Ön İşleme
* **Veri Seti:** 4.350 benzersiz hava/uydu fotoğrafı (`nofire`: 2.410, `fire`: 1.940).
* **Veri Pipeline:** `pathlib.Path` ile OS bağımsız mimari, `pin_memory=True` ve `num_workers=2` ile VRAM darboğaz engelleme optimizasyonları.
* **Augmentation (Veri Artırma):** Değişken duman yönleri ve simetrik ışık sahneleri için `RandomHorizontalFlip()` entegrasyonu.

### 2. Derin Öğrenme Motoru (EfficientNet-B0)
Modelin orijinal ImageNet (1000 sınıflı) çıkış katmanı kesilmiş, yerine projenin doğasına uygun olan 2 sınıflı yeni bir doğrusal katman örülmüştür:
```python
model.classifier[1] = nn.Linear(num_features, 2)


Sınıflar: 0 -> nofire (Sağlıklı Orman), 1 -> fire (Yangın/Duman)
```
### ⚙️ Hiperparametreler 
* **Kayıp Fonksiyonu (Loss):** `CrossEntropyLoss` (Sınıf olasılık dağılımlarını ölçmek için)
* **Optimizasyon Algoritması:** `AdamW` (Ağırlık sönümlemeli gradyan sabitleyici)
* **Öğrenme Oranı (Learning Rate):** learning_rate = 1e-4  # 1 * 10^-4 (0.0001)
* **Paket Boyutu (Batch Size):** `32`

---

## 📊 Eğitim Performans Analizi

Yapay zeka motoru, 5 adımlı eğitim döngüsü (*Tezgahı temizle* $\rightarrow$ *Tahmin et* $\rightarrow$ *Hatayı ölç* $\rightarrow$ *Sorumluyu bul* $\rightarrow$ *Vidaları sık*) eşliğinde sadece 3 epoch içerisinde muazzam bir öğrenme eğrisi yakalamıştır:

| Dönem (Epoch) | Ortalama Kayıp (Loss) | Eğitim Doğruluğu (Train Acc) |
| :---: | :---: | :---: |
| **Epoch 1** | 0.1746 | %99.62 |
| **Epoch 2** | 0.0411 | %99.72 |
| **Epoch 3** | **0.0164** | **%99.84** |

### ✅ Görünmeyen Veri Testi (Validation)
Modelin eğitim esnasında hiç görmediği ve ezber testi yaptığı doğrulama setindeki nihai karne şu şekildedir:
* 🎯 **Validation Accuracy (Test Başarısı):** `%97.40`
* 📉 **Validation Loss (Test Kaybı):** `0.2160`

> 📌 **Mühendis Değerlendirmesi:** Eğitim doğruluk skorunun (%99.84) ve test doğruluk skorunun (%97.40) birbirine son derece yakın ilerlemesi; modelde bir ezberleme (**Overfitting**) durumu oluşmadığının, modelin orman ve yangın dokusunu gerçekten kavradığının kesin kanıtıdır.

---

## 🛰️ NASA FIRMS Entegrasyonu ve Yerelleştirme

### 🌡️ Kelvin ➔ Celsius Dönüşümü
NASA sunucularından gelen ham bilimsel veriler Kelvin ($K$) cinsindendir. Sahadaki operasyon ekiplerinin durumu saniyeler içinde doğru yorumlayabilmesi adına veriler şu mühendislik formülüyle yerel standartlara dönüştürülmüştür:

 celsius = kelvin - 273.15
   

*Bu dönüşüm sayesinde haritadaki karmaşık endüstriyel değerler, anlaşılır sıcaklık göstergelerine (°C) evrilmiştir.*

### 🗺️ GIS Coğrafi İşaretleme
Sistem coğrafi olarak `folium` ve `streamlit-folium` kütüphaneleriyle görselleştirilmiştir. NASA simülatöründen gelen aktif ısıl anomali noktaları Türkiye coğrafi sınırları içerisinde filtrelenerek harita üzerine kırmızı işaretçiler (**Markers**) olarak jilet gibi konumlandırılmıştır.

---

## 💾 Model Depolama ve Sınır Bilişim (Edge AI) Uyumluluğu

Eğitilen yapay zeka beyninin öğrendiği tüm katsayılar (weights), PyTorch standartlarında diske kilitlenmiştir:
* 📦 **Nihai Dosya Boyutu:** `15.59 MB` (`efficientnet_b0_forest_fire.pt`)

`15.59 MB` boyutundaki bu mikro ağırlık dosyası, modelin doğruluğundan ödün vermeksizin elde edilmiş bir mühendislik başarısıdır. Bu hafiflik sayesinde model; internet kısıtı olan orman gözlem kulelerindeki mikrodenetleyicilere, otonom dronlara ve mobil cihazlara bulut sunucu bağımlılığı olmadan (**Offline / On-Device**) gömülebilecek seviyededir.

---

## 🚀 Gelecek Vizyonu — Proje Yol Haritası

Bu çalışma, sertifika sürecinin ardından hayata geçirilecek olan geniş kapsamlı profesyonel çevresel veri analizi uygulamalarının çekirdek mekanizmasını oluşturmaktadır.

**Gelecek Fazlarda Eklenecek Yapılar:**
* 🛰️ **Sentinel-2 Canlı Uydu API:** NASA'dan gelen koordinat bilgisiyle eş zamanlı olarak o koordinatın en güncel Sentinel uydu fotoğrafı internetten otomatik indirilip eğittiğimiz EfficientNet modeline paslanacaktır.
* 🚨 **Otonom Alarm & Drone Yönlendirme:** Yangın doğrulandığı an bölgedeki otonom dronlara rota ataması yapılacak ve Edge TPU (Google Coral) donanım hızlandırıcıları ile saniyede 30+ kare görüntü işleme (*Real-time Inference*) seviyesine çıkılacaktır.

---

## 📋 Yerel Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza klonlayın:
```bash
git clone [https://github.com/kullanici_adi/orman-yangini-takip.git](https://github.com/kullanici_adi/orman-yangini-takip.git)
cd orman-yangini-takip
```
2. Gerekli kütüphaneleri (malzemeleri) tedarik edin:
```bash
pip install -r requirements.txt
```
3. Uygulamayı yerelde ayağa kaldırın:
```bash
streamlit run app.py
```
---

<br>

<table align="center" style="border: none; border-collapse: collapse;">
  <tr>
    <td align="center" style="border: none; padding: 20px;">
      <p align="center"><b>Hazırlayan</b></p>
      <h2 align="center" style="color: #2a98a4; margin-top: 5px;">📐 Serdar ÖNAL</h2>
      <p align="center" style="margin-bottom: 2px;"><b>Kıdemli İnşaat Mühendisi & Yapay Zeka Araştırmacısı</b></p>
      <p align="center" style="color: #4A778D; font-style: italic; font-size: 0.9em;">"Veri ile İnşa Edilen Sürdürülebilir ve Güvenli Bir Gelecek İçin..."</p>
      <p align="center" style="font-size: 0.8em; color: #888888; margin-top: 15px;">© 2026 | Edge-AI-16 Altyapı ve Yapay Zeka Projesi</p>
    </td>
  </tr>
</table>