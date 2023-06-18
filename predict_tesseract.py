import os
import time

import serial
from ultralytics import YOLO
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import string
from unidecode import unidecode

arduino_port = "COM4"  # Arduino'nun bağlı olduğu seri portu belirtin
baud_rate = 9600
arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
try:
    if arduino.is_open:
        print("Seri port zaten açık")
        arduino.close()  # Seri portu kapat
    else:
        print("Seri port açıldı")
    
    arduino.open()  # Seri portu tekrar aç
    if arduino.is_open:
        print("Seri port tekrar açıldı")
        # Arduino ile iletişime geçmek için diğer işlemleri burada gerçekleştirin
        arduino.write(b"1")  # Arduino'ya komut gönder
    else:
        print("Seri port açılamadı")
except serial.SerialException as e:
    print(f"Seri port açılırken hata oluştu: {e}")

cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
offset = 17 #plakanın sınırlarını sol taraftan daraltmak için kullanılan değişken
text_final = ""
cache = ""

ainput_dir = os.path.join("data")

adata_path = os.path.join("data", "001.jpg") 

current_dir = os.getcwd()

input_dir = os.path.join(current_dir, ainput_dir)
data_path = os.path.join(current_dir, adata_path)

while True:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()

    
    # Görüntüyü belirtilen yol üzerindeki dosyaya yaz
    cv2.imwrite(data_path, frame)
    
    img = cv2.imread(data_path)

    if img is None:
        print("Görüntü yüklenemedi")
        continue  # Bir sonraki döngüye geç
    
    #blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
    
    results = model.predict(source=data_path, show=False)

    
    bboxes = []

    for result in results:
        if len(result.boxes.xywh) > 0:
            bbox = result.boxes.xywh[0]
            xc, yc, w, h = bbox.tolist()
            bbox_confidence = result.boxes.conf[0]
            bboxes.append(bbox)
        else:
            print("No license plate detected")
            continue

    #reader = easyocr.Reader(['en'])

    for bbox_, bbox in enumerate(bboxes):
        
        xc, yc, w, h = bbox
        
        #xc -= offset/2
        
        #w += 2 * offset
        #h += 2 * offset

        license_plate = img[int(yc - (h / 2)) + 4:int(yc + (h / 2)), int(xc - (w / 2)) + offset:int(xc + (w / 2)) - 4, :].copy()




        if license_plate is not None:
            
            license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

            Ha, Wa = license_plate_gray.shape

            d = cv2.resize(license_plate_gray, (Wa*10, Ha*10))

            blurred_image = cv2.GaussianBlur(d, (9,9), 0)

            histogram_e = cv2.equalizeHist(blurred_image)

            Ha, Wa = d.shape
            
            g = cv2.resize(histogram_e, (int(Wa/10), int(Ha/10)))

            _, license_plate_thresh = cv2.threshold(license_plate_gray, 55, 255, cv2.THRESH_BINARY_INV)
            
            #cv2.imshow("License Plate", license_plate_thresh)

            data = pytesseract.image_to_data(license_plate_thresh, output_type='data.frame')

            filtered_data = data[data['conf'] > 20]

            plate_text = ' '.join(str(value) for value in filtered_data['text'].values)

            latin = unidecode(plate_text)

            remove_punctuation = latin.translate(str.maketrans('', '', string.punctuation)) 

            remove_space = remove_punctuation.replace(" ", "")

            def format_string(input_string):
                components = []
                current_component = ""

                for char in input_string:
                    if char.isdigit():
                        if not components:
                            current_component += char
                        else:
                            components.append(current_component)
                            current_component = char
                    elif char.isalpha():
                        if not current_component and not components:
                            continue
                        elif not current_component and components:
                            components.append(char)
                        elif current_component.isdigit():
                            components.append(current_component)
                            current_component = char
                        else:
                            current_component += char

                if current_component:
                    components.append(current_component)

                # Başta ilk sayıya kadar olan harfi sil
                if components and components[0].isalpha():
                    components = components[1:]

                # Son sayıdan en sona kadar olan harfi sil
                if components and components[-1].isalpha():
                    components = components[:-1]

                formatted_string = ' '.join(components)
                return formatted_string

            
            
            #def validate_data(data):
            def kontrol_et(veri):
                # Verinin uzunluğunu kontrol et
                if len(veri) < 3:
                    return False

                # İlk karakter sayı mı diye kontrol et
                if not veri[0].isdigit():
                    return False

                # Son karakter sayı mı diye kontrol et
                if not veri[-1].isdigit():
                    return False

                # Harf ve sayıları sayacak değişkenler
                harf_sayisi = 0
                sayi_sayisi = 0

                # Verinin karakterlerini kontrol et
                for karakter in veri[1:-1]:
                    if karakter.isalpha():
                        harf_sayisi += 1
                    elif karakter.isdigit():
                        sayi_sayisi += 1
                    else:
                        return False  # Geçersiz karakter bulundu

                # En az bir harf ve bir sayı olmalı
                if harf_sayisi == 0 or sayi_sayisi == 0:
                    return False

                # Eğer yukarıdaki kontrolleri geçtiyse, doğru bir kombinasyon olduğunu belirt
                return True

                    
            dezenlenmis = format_string(remove_space)

            remove_space2 = dezenlenmis.replace(" ", "")
            dogrulama = kontrol_et(remove_space2)

            
            if dogrulama == True:
                if remove_space2 == cache:
                    print("Aynı plaka")
                    print(remove_space2)
                else:
                    print("Plaka doğrulandı")
                    text_final = remove_space2
                    print(plate_text)
                    print(remove_space)
                    print(dezenlenmis)
                    print(remove_space2)
                    print(dogrulama)
                    print(text_final)
                    if text_final == "16RAM14" or "16SBL55":
                        
                        arduino.write(b"1")
                        time.sleep(1) 
                        arduino.write(b"0")
                    cache = text_final

            
        else:
            print("No license plate detected")

    if cv2.waitKey() & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
