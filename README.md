# Türk Plaka Okuyucu
Türk plakalarını algılar okur ve yazı halinde çıktı verir 

*Bu proje kişisel bir projedir, tamamen ben tarafından kişisel gelişim ve öğrenim amacıyla hazırlanmıştır.*
# Ana dosya (Windows):
[Predict_Windows_1.0.py](https://github.com/emrs6/TurkPlakaOkuyucu/blob/main/Predict_Windows_1.0.py)

# İşlemler :
  -Kendi yapyığım 600 plakalık ufak bir YOLOV8n kütüphanesi ile plakaları tanımlar<br/>
  -Plaka bölümünü kırpar<br/>
  -TesseractOCR / EasyOCR ile yazıyı çıkartır<br/>

# Gerekli kütüphaneler (Windows sürümü için):
  -PyThorch(CUDA)<br/>
  -time<br/>
  -cv2<br/>
  -easyocr<br/>
  -numpy<br/>
  -pandas<br/>
  -ultralytics<br/>
  -requests<br/>
  -matplotlib<br/>
>   Model https://drive.google.com/file/d/1CNQLLDu-SFhca1vGdjzTmwQoXcpq9ppQ/view?usp=sharing<br/>

# Çalışma Mantığı ve detaylar:
  *Bilgisayarınızın hızına göre değişiklik gösteren bir hıza sahiptir*
  1) Kamera üzerinden bir fotoğraf çekilir
  2) Fotoğrafta YOLO modülü ile araç aranır
  3) Bulunan araçta YOLO modülü ile plaka aranır
  4) Bulunan plakanın konumu alınır
  5) Plakanın olduğu yerler kırpılır
  6) Kırpılan resimde işlem yapılır (Siyah beyaz yapma ve Thresh):
  7) Kırpılmış plaka üzerinde tesseract-ocr/easyocr çalıştırılır
  8) Çıkan sonuçta filtreleme yapılır:<br/>
    A) Yanlışlıkla okunan noktalama işaretleri silinir<br/>
    B) Çıkan kombinasyonun -sayı-harf-sayı- şeklinde olduğu kontrol edilir:<br/>
       Örnek: B16XYZ123 şeklinde okunan plaka 16XYZ123 şekline dönüştürülür<br/>
    C) Plakadakı boşluklar silinir<br/>
  9) Çıktı terminale yazdırılır

# IFTTT Webhook ayarı (Raspberry Pi sürümleri):
  Predict_3_x dosyaları, izinli plaka okunduğunda IFTTT Webhooks ile kapıyı tetikler. Webhook anahtarı koda yazılmaz, ortam değişkeninden okunur:<br/>
  1) `.env.example` dosyasını `.env` adıyla kopyalayın ve `IFTTT_WEBHOOK_KEY` değerini doldurun<br/>
  2) `.env` dosyasının otomatik okunması için `pip install python-dotenv` kurun (ya da değişkeni terminalde `export IFTTT_WEBHOOK_KEY=...` ile tanımlayın)<br/>
  3) `IFTTT_EVENT` boş bırakılırsa `door_trigger` kullanılır<br/>
  4) Koddaki örnek plakayı (`16XYZ123`) kendi izinli plakanızla değiştirin<br/>
  *`IFTTT_WEBHOOK_KEY` tanımlı değilse script çalışmaya devam eder, yalnızca webhook çağrısı atlanır. `.env` dosyasını asla commit etmeyin.*
