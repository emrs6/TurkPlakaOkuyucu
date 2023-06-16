# Türk Plaka Okuyucu
Türk plakalarını algılar okur ve yazı halinde çıktı verir 

*Bu proje kişisel bir projedir ve tamamen ben tarafından hazırlanmıştır.*

# İşlemler :
  -Kendi yapyığım 600 plakalık ufak bir YOLOV8n kütüphanesi ile plakaları tanımlar
  -Plaka bölümünü kırpar
  -TesseractOCR ile yazıyı çıkartır

# Gerekli kütüphaneler:
  -numpy
  -cv2
  -pytesseract
  -unidecode
  -io
  -ultralytics

# Çalışma Mantığı ve detaylar:
  *Bilgisayarınızın hızına göre değişiklik gösteren bir hıza sahiptir*
  1) Kamera üzerinden bir fotoğraf çekilir
  2) Fotoğrafta YOLO modülü ile plaka aranır
  3) Bulunan plakanın konumu alınır
  4) Plakanın olduğu yerler kırpılır
  5) Kırpılan resimde işleme yapılır (Siyah beyaz yapma ve Thresh):
  
    ![image](https://github.com/emrs6/TurkPlakaOkuyucu/assets/65279699/b4d03317-2611-4620-8ace-8403c560efe5)

  6) Kırpılmış plaka üzerinde tesseract-ocr çalıştırılır
  7) Çıkan sonuçta filtreleme yapılır:
    A) Yanlışlıkla okunan noktalama işaretleri silinir
    B) Çıkan kombinasyonun -sayı-harf-sayı- şeklinde olduğu kontrol edilir:
       Örnek: B16ACL82 şeklinde okunan plaka 16ACL82 şekline dönüştürülür
    C) Plakadakı boşluklar silinir
  8) Çıktı terminale yazdırılır
  
# Daha sonra yapılacaklar:
  1) SQL database ile bağlantı kurulup plakanın sahaya giriş saati ve çıkış saati yazdırılacak
  2) Whitelist ile sadece belli plakalar geldiğinde otomatik kapı açılacak

# Yapılabilecek şeyler
  -Custom OCR
