# Türk Plaka Okuyucu
Türk plakalarını algılar okur ve yazı halinde çıktı verir 

*Bu proje kişisel bir projedir ve tamamen ben tarafından hazırlanmıştır.*
# Ana dosya (Windows):
[Predict_Windows_1.0.py](https://github.com/emrs6/TurkPlakaOkuyucu/blob/main/Predict_Windows_1.0.py)

# İşlemler :
  -Kendi yapyığım 600 plakalık ufak bir YOLOV8n kütüphanesi ile plakaları tanımlar<br/>
  -Plaka bölümünü kırpar<br/>
  -TesseractOCR / EasyOCR ile yazıyı çıkartır<br/>

# Gerekli kütüphaneler (Winows sürümü için):
  -os<br/>
  -time<br/>
  -cv2<br/>
  -easyocr<br/>
  -numpy<br/>
  -pandas<br/>
  -ultralytics<br/>
  -requests<br/>
  -string<br/>
  -matplotlib<br/>
>   Model https://drive.google.com/file/d/1CNQLLDu-SFhca1vGdjzTmwQoXcpq9ppQ/view?usp=sharing<br/>

# Çalışma Mantığı ve detaylar:
  *Bilgisayarınızın hızına göre değişiklik gösteren bir hıza sahiptir*
  1) Kamera üzerinden bir fotoğraf çekilir
  2) Fotoğrafta YOLO modülü ile plaka aranır
  3) Bulunan plakanın konumu alınır
  4) Plakanın olduğu yerler kırpılır
  5) Kırpılan resimde işlem yapılır (Siyah beyaz yapma ve Thresh):
  6) Kırpılmış plaka üzerinde tesseract-ocr çalıştırılır
  7) Çıkan sonuçta filtreleme yapılır:<br/>
    A) Yanlışlıkla okunan noktalama işaretleri silinir<br/>
    B) Çıkan kombinasyonun -sayı-harf-sayı- şeklinde olduğu kontrol edilir:<br/>
       Örnek: B16ACL82 şeklinde okunan plaka 16ACL82 şekline dönüştürülür<br/>
    C) Plakadakı boşluklar silinir<br/>
  8) Çıktı terminale yazdırılır
  
# Daha sonra yapılacaklar:
  1) SQL database ile bağlantı kurulup plakanın sahaya giriş saati ve çıkış saati yazdırılacak
  2) Whitelist ile sadece belli plakalar geldiğinde otomatik kapı açılacak


#Uzaktan kontrol linki:
  https://maker.ifttt.com/trigger/door_trigger/json/with/key/45MZeHwhX454RLs1GGu6d
