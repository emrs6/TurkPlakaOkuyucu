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
  -Kamera üzerinden bir fotoğraf çekilir
  -Fotoğrafta YOLO modülü ile plaka aranır
  -Bulunan plakanın konumu alınır
  -Plakanın olduğu yerler kırpılır
  -Kırpılan resimde işleme yapılır (Siyah beyaz yapma ve Thresh):
  
    ![image](https://github.com/emrs6/TurkPlakaOkuyucu/assets/65279699/b4d03317-2611-4620-8ace-8403c560efe5)

  -Kırpılmış plaka üzerinde tesseract-ocr çalıştırılır
  -Çıkan sonuçta filtreleme yapılır:
    -Yanlışlıkla okunan noktalama işaretleri silinir
    -Çıkan kombinasyonun -sayı-harf-sayı- şeklinde olduğu kontrol edilir
       Örnek: B16ACL82 şeklinde okunan plaka 16ACL82 şekline dönüştürülür
    -Plakadakı boşluklar silinir
  -Çıktı terminale yazdırılır
  
# Daha sonra yapılacaklar:
  1) SQL database ile bağlantı kurulup plakanın sahaya giriş saati ve çıkış saati yazdırılacak
  2) Whitelist ile sadece belli plakalar geldiğinde otomatik kapı açılacak

# Yapılabilecek şeyler
  -Custom OCR
