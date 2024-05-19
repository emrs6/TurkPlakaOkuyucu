# Türk Plaka Okuyucu
Türk plakalarını algılar okur ve yazı halinde çıktı verir 

*Bu proje kişisel bir projedir ve tamamen ben tarafından hazırlanmıştır.*
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
       Örnek: B16ACL82 şeklinde okunan plaka 16ACL82 şekline dönüştürülür<br/>
    C) Plakadakı boşluklar silinir<br/>
  9) Çıktı terminale yazdırılır
