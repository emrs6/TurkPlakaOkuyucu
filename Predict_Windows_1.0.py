#Bu sürüm Windows bilgisayarlarda kullanım için oluşturuldu. GPU hızlandırıcı kullanır numpy gpu cuda sürümü gerekir 

 #import I2C_LCD_driver 
from time import *
import cv2
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO
import requests
import matplotlib.pyplot as plt

print("kütüphaneler yüklendi")

cap = cv2.VideoCapture("test.mp4")
model = YOLO("best.pt")
car_model = YOLO("yolov8n.pt")
model.to('cuda')
car_model.to('cuda')
vehicles = [2, 3, 5, 7]
offset = 17  # plakanın sınırlarını sol taraftan daraltmak için kullanılan değişken
text_final = ""
cache = ""
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)
reader = easyocr.Reader(['en'])
print("parametreler tanımlandı")

# Definitions

def trigger_webhook(webhook_url):
    try:
        response = requests.post(webhook_url)
        if response.status_code == 200:
            print("Webhook successfully triggered.")
            print("Response content:")
            print(response.text)  # Yanıttaki içeriği yazdır
        else:
            print(f"Failed to trigger webhook. Status code: {response.status_code}")
            print("Response content:")
            print(response.text)  # Yanıttaki içeriği yazdır
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending the webhook request: {e}")

def str_to_nmb(text):
    
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("G", "6")
    text = text.replace("Z", "2")
    text = text.replace("z", "2")
    text = text.replace("I", "1")
    text = text.replace("ı", "1")
    text = text.replace("o", "0")
    text = text.replace("O", "0")
    text = text.replace("S", "5")
    text = text.replace("s", "5")
    text = text.replace("L", "4")
    text = text.replace("l", "1")
    text = text.replace("D", "0")
    text = text.replace("B", "8")
    return text


# işleme modülü 2.0
def process2_1(imgraw):
    Ha, Wa = imgraw.shape[:2]
    img = cv2.resize(imgraw, (Wa * 1, Ha * 1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    denoised = cv2.blur(denoised, (5, 5))
    thresh = cv2.threshold(denoised,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=5)
    return (erosion, img)


def resize_image_width(image, target_width):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image


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

def alternfunc(PlateImg):
    plt.imshow(PlateImg)
    plt.show()
    if not PlateImg.shape[1] >= 85:
        print("Plaka boyutları alternatif işleme için çok küçük")
    else:
        altern_img_resized_ = resize_image_width(PlateImg, 1750)
        altern_gray_letter = cv2.cvtColor(altern_img_resized_, cv2.COLOR_BGR2GRAY)
        altern_noise_reduce_letter = cv2.bilateralFilter(altern_gray_letter, 9, 75, 75)
        altern_noise_reduce_letter = cv2.blur(altern_noise_reduce_letter, (5, 5))
        altern_thresholed_letter = cv2.threshold(altern_noise_reduce_letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        altern_thresholed_letter = cv2.bitwise_not(altern_thresholed_letter)
        altern_thresholed_letter = cv2.cvtColor(altern_thresholed_letter, cv2.COLOR_GRAY2BGR)
        plt.imshow(altern_thresholed_letter)
        plt.show()
        blocklist = '!"#%&\'()*+,-.:;<=>?@[]^_`{|}~ '
        altern_data = reader.readtext(altern_thresholed_letter, paragraph=True, blocklist=blocklist)
        altern_data = sorted(altern_data, key=lambda x: x[0][0][0])
        altern_sonuclar = [item[1] for item in altern_data]
        if altern_sonuclar is not None and len(altern_sonuclar) != 0:
            altern_plaka_verisi = str(altern_sonuclar[0])
        else:
            altern_plaka_verisi = ""
        print("veri"+altern_plaka_verisi)
        if len(altern_plaka_verisi) > 5:

            # sonuç düzenleme

            print("Çıkarılan plaka:")
            altern_plaka = str(altern_sonuclar[0])

            altern_plaka = altern_plaka.replace("$", "S")
            altern_plaka = altern_plaka.replace("/", "I")
            altern_plaka = altern_plaka.replace(" ", "")
            altern_plaka = format_string(altern_plaka)
            if len(altern_plaka) > 5:
                
                alan_kodu = altern_plaka[:2]
                geri_kalan = altern_plaka[2:-2]
                son2rakam = altern_plaka[-2:]
                print("İşleme verilerine ayrılıyor:")
                print("Alan kodu:")
                print(alan_kodu)
                print("Orta bölüm:")
                print(geri_kalan)
                print("Son iki rakam:")
                print(son2rakam)
                print("işleniyor...")

                alan_kodu = str_to_nmb(alan_kodu)

                if not alan_kodu[0].isalpha():
                    harf_3 = geri_kalan[0]
                    other_cache = geri_kalan[1:]
                    harf_3 = harf_3.replace(" ", "")
                    harf_3 = harf_3.replace("6", "G")
                    harf_3 = harf_3.replace("2", "Z")
                    harf_3 = harf_3.replace("1", "I")
                    harf_3 = harf_3.replace("0", "O")
                    harf_3 = harf_3.replace("5", "S")
                    harf_3 = harf_3.replace("4", "L")
                    harf_3 = harf_3.replace("8", "B")

                    geri_kalan = harf_3 + other_cache
                    geri_kalan = geri_kalan.upper()
                else:
                    geri_kalan = geri_kalan.upper()

                son2rakam = str_to_nmb(son2rakam)

                for _ in range(5):
                    geri_kalan_01 = ""

                    for i in range(len(geri_kalan)):
                        harf = geri_kalan[i]

                        if harf == "O":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakterO = geri_kalan[i - 1]
                                sag_karakterO = geri_kalan[i + 1]

                                if not (sol_karakterO.isalpha() and sag_karakterO.isalpha()):
                                    geri_kalan_01 += "0"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "0"

                        elif harf == "I":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakterI = geri_kalan[i - 1]

                                sag_karakterI = geri_kalan[i + 1]

                                if not (sol_karakterI.isalpha() and sag_karakterI.isalpha()):
                                    geri_kalan_01 += "1"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "1"

                        elif harf == "0":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter0 = geri_kalan[i - 1]
                                sag_karakter0 = geri_kalan[i + 1]

                                if sol_karakter0.isalpha() and sag_karakter0.isalpha():
                                    geri_kalan_01 += "O"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "0"

                        elif harf == "1":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter1 = geri_kalan[i - 1]
                                sag_karakter1 = geri_kalan[i + 1]

                                if sol_karakter1.isalpha() and sag_karakter1.isalpha():
                                    geri_kalan_01 += "I"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "1"

                        elif harf == "2":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter2 = geri_kalan[i - 1]
                                sag_karakter2 = geri_kalan[i + 1]

                                if sol_karakter2.isalpha() and sag_karakter2.isalpha():
                                    geri_kalan_01 += "Z"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "2"

                        elif harf == "4":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter4 = geri_kalan[i - 1]
                                sag_karakter4 = geri_kalan[i + 1]

                                if sol_karakter4.isalpha() and sag_karakter4.isalpha():
                                    geri_kalan_01 += "l"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "4"

                        elif harf == "5":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter5 = geri_kalan[i - 1]
                                sag_karakter5 = geri_kalan[i + 1]

                                if sol_karakter5.isalpha() and sag_karakter5.isalpha():
                                    geri_kalan_01 += "S"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "5"

                        elif harf == "6":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter6 = geri_kalan[i - 1]
                                sag_karakter6 = geri_kalan[i + 1]

                                if sol_karakter6.isalpha() and sag_karakter6.isalpha():
                                    geri_kalan_01 += "G"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "6"

                        elif harf == "8":
                            # Solunda ve sağındaki karakterlerin kontrolü
                            if 0 < i < len(geri_kalan) - 1:
                                sol_karakter8 = geri_kalan[i - 1]
                                sag_karakter8 = geri_kalan[i + 1]

                                if sol_karakter8.isalpha() and sag_karakter8.isalpha():
                                    geri_kalan_01 += "B"
                                else:
                                    geri_kalan_01 += harf
                            else:
                                geri_kalan_01 += "8"

                        else:
                            geri_kalan_01 += harf

                    geri_kalan = geri_kalan_01

                altern_plaka = alan_kodu + geri_kalan + son2rakam
                print(altern_plaka)
                formatted_plate = format_string(altern_plaka)
                formatted_plate = formatted_plate.replace(" ", "")
                print(formatted_plate)

                print("İşleme tamam. Doğruluk kontrol ediliyor...")
                check_plate = kontrol_et(formatted_plate)
                if check_plate:
                    print("Plaka Kombinasyonu doğru")
                    print("Tahmini plaka:")
                    print(formatted_plate)
                    #mylcd.lcd_clear()
                    #mylcd.lcd_display_string(formatted_plate, 1)

                else:
                    print("doğrulanamadı")
                    #mylcd.lcd_clear()
                    #mylcd.lcd_display_string("hata", 1)
            else:
                print("okuma başarısız")
                
                # =====================================================================#
# Kamera ile foto çek
while True:
    ret, frame = cap.read()
    if frame is None:
        print("Görüntü yüklenemedi")
        continue  # Bir sonraki döngüye geç
    else:
        img = frame.copy()
    reduced_image = resize_image_width(img, 768)

    car_detections = car_model(source=reduced_image, show=False)[0]

    car_detections_ = []
    for detection in car_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            x1 = x1 * 6
            y1 = y1 * 6
            x2 = x2 * 6
            y2 = y2 * 6
            
            car_detections_.append([x1, y1, x2, y2, score])
        else:
            print("araç algılanamadı")


    for car_detection in car_detections_:
        print(car_detections_[0])
        xcar, ycar, x2car, y2car, score = car_detection
        if score > 0.4:
            car = img[int(ycar):int(y2car), int(xcar):int(x2car), :].copy()
               
            if car is None or car.size == 0:
                continue
            car = resize_image_width(car, 1750)
            # plakayı bul
            results = model.predict(car, show=False)
            bboxes = []
            for result in results:
                if len(result.boxes.xywh) > 0:
                    bbox = result.boxes.xywh[0]
                    xc, yc, w, h = bbox.tolist()
                    bbox_confidence = result.boxes.conf[0]
                    bboxes.append(bbox)
                    print("plaka değişkenleri atandı")
                    

                    license_plate = None
                    # plakayı kırp ve kaydet
                    for bbox_, bbox in enumerate(bboxes):
                        xc, yc, w, h = bbox
                        offset = 10
                        w = w + 15
                        h = h + 15
                        license_plate = car[int(yc - (h / 2)):int(yc + (h / 2)),
                                        int(xc - (w / 2)) + offset:int(xc + (w / 2)), :].copy()
                        yolo = license_plate.copy()
                        alternPredict = license_plate.copy()
                        

                    imgraw = yolo.copy()
                    
                    if not imgraw.shape[1] >= 85:
                        print("Plaka boyutları çok küçük")

                        continue
                    if imgraw.shape[0] < 10:
                        continue
                    img_resized_ = resize_image_width(imgraw, 1750)

                    erosion = process2_1(img_resized_)[0]
                    img = process2_1(img_resized_)[1]

                    # Dikdörtgen (kontür) bulma işlemi
                    cropped_contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Kontür bilgilerini içeren bir DataFrame oluşturma
                    data = []
                    for i, cnt in enumerate(cropped_contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        aspect_ratio = w / h
                        data.append([i + 1, x, y, w, h, area, aspect_ratio])

                    df = pd.DataFrame(data, columns=['Sıra No', 'x', 'y', 'width', 'height', 'area', 'aspect_ratio'])
                    max_area_row = df[df['area'] == df['area'].max()]
                    areaNmb = max_area_row['Sıra No'].values[0]
                    areax = max_area_row['x'].values[0]
                    areay = max_area_row['y'].values[0]
                    areawidth = max_area_row['width'].values[0]
                    areaheight = max_area_row['height'].values[0]
                    print("En büyük dikdörtgen değerleri:")
                    print("Sıra No:", areaNmb)
                    print("x:", areax)
                    print("y:", areay)
                    print("width:", areawidth)
                    print("height:", areaheight)
                    # DataFrame'i ekrana yazdırma
                    #print(df)
                    # en büyük dikdörtgeni kırpma
                    license_plate_cropped = img[areay:areay + areaheight, areax:areax + areawidth].copy()

                    # kırpılan dikdörtgene harf çıkarma için filtre
                    cropped_gray = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
                    cropped_denoised = cv2.bilateralFilter(cropped_gray, 9, 75, 75)
                    cropped_denoised = cv2.blur(cropped_denoised, (5, 5))
                    cropped_thresh = cv2.threshold(cropped_denoised,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                    cropped_erosion = cv2.bitwise_not(cropped_thresh)

                    # harfleri bulma
                    letters, hierarchy = cv2.findContours(cropped_erosion, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    # intgernal kontürleri bulmak için RETR_LIST veya RETR_CCOMP kullanmak gerekiyor
                    # letters, _ = cv2.findContours(bgcr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    print(hierarchy)

                    letterData = []
                    # eğer en boy oranı 0.20 ile 0.90 arasında ise ve alanı 9000 den büyük ise harf olarak değişkene al
                    for i, cnt in enumerate(letters):
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        aspect_ratio = w / h
                        if (0.20 <= aspect_ratio <= 0.90) and (area > 9000) and (hierarchy[0][i][3] == -1):
                            letterData.append([x, y, w, h])

                    letterData = sorted(letterData, key=lambda box: box[0])
                    # print(letterData)

                    # harflere alan ayarlama
                    letters = []
                    for i, box in enumerate(letterData):
                        x, y, w, h = box
                        # harf resmi işleme

                        gray_letter = cv2.cvtColor(license_plate_cropped[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

                        noise_reduce_letter = cv2.bilateralFilter(gray_letter, 9, 75, 75)
                        noise_reduce_letter = cv2.blur(noise_reduce_letter, (5, 5))
                        #contrast_letter = cv2.convertScaleAbs(noise_reduce_letter, alpha=2, beta=0)
                        
                        thresholed_letter = cv2.threshold(noise_reduce_letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                        

                        thresholed_letter = cv2.bitwise_not(thresholed_letter)
                        thresholed_letter = cv2.cvtColor(thresholed_letter, cv2.COLOR_GRAY2BGR)
                        # harfin koyulacağı alan boyutu (kendi boyutundan 1.44 kat daha büyük)
                        bgH, bgW = thresholed_letter.shape[:2]
                        bg = np.zeros((int(bgH * 1.2), int(bgW * 1.2), 3)
                        , dtype=np.uint8)
                        bgY_offset = int((bg.shape[0] - bgH) / 2)
                        bgX_offset = int((bg.shape[1] - bgW) / 2)

                        bg[bgY_offset:bgY_offset + bgH, bgX_offset:bgX_offset + bgW] = thresholed_letter

                        kernel = np.ones((5, 5), np.uint8)
                        erosion = cv2.erode(bg, kernel, iterations=1)

                        # işlenen harfi kaydetme
                        letters.append(erosion)

                    if letters and not all(image is None for image in letters):
                        # Harfleri yanyana koyarak oluşturacağımız yeni plaka için genişlik ve yükseklik verileri
                        max_height = max(image.shape[0] for image in letters)
                        total_length = sum(image.shape[1] for image in letters)

                        combined_image = np.zeros((max_height, total_length, 3), dtype=np.uint8)

                        x_offset = 0
                        for image in letters:
                            h_letter, w_letter = image.shape[:2]
                            combined_image[0:h_letter, x_offset:x_offset + w_letter] = image
                            x_offset += w_letter

                        # ===================Okuma bölümü=======================#
                        # Çerçeve oluşturma
                        hcom, wcom = combined_image.shape[:2]
                        black_rectangle = np.zeros((int(hcom * 2), int(wcom * 2), 3), dtype=np.uint8)
                        ycom_offset = int((black_rectangle.shape[0] - hcom) / 2)
                        xcom_offset = int((black_rectangle.shape[1] - wcom) / 2)
                        black_rectangle[ycom_offset:ycom_offset + hcom, xcom_offset:xcom_offset + wcom] = combined_image

                        hcom, wcom = black_rectangle.shape[:2]
                        black_rectangle = cv2.resize(black_rectangle, (wcom // 4, hcom // 4))
                        hcom, wcom = black_rectangle.shape[:2]
                        black_rectangle2 = np.zeros((int(hcom * 1.3), int(wcom * 1.3), 3), dtype=np.uint8)
                        ycom_offset = int((black_rectangle2.shape[0] - hcom) / 2)
                        xcom_offset = int((black_rectangle2.shape[1] - wcom) / 2)
                        black_rectangle2[ycom_offset:ycom_offset + hcom,
                        xcom_offset:xcom_offset + wcom] = black_rectangle
                        
                        black_rectangle2 = resize_image_width(black_rectangle2 , 300)
                        

                        reversed_rct = cv2.bitwise_not(black_rectangle2)

                        print("okunuyor")
                        # okuma
                        blocklist = '!"#%&\'()*+,-.:;<=>?@[]^_`{|}~ '
                        data = reader.readtext(reversed_rct, paragraph=True, blocklist=blocklist)
                        # x verisine göre sıralama
                        data = sorted(data, key=lambda x: x[0][0][0])
                        sonuclar = [item[1] for item in data]
                        if sonuclar is not None and len(sonuclar) != 0:
                            plaka_verisi = str(sonuclar[0])
                        else:
                            plaka_verisi = ""
                        print(plaka_verisi)
                        if len(plaka_verisi) > 5:

                            # sonuç düzenleme

                            print("Çıkarılan plaka:")
                            plaka = str(sonuclar[0])

                            plaka = plaka.replace("$", "S")
                            plaka = plaka.replace("/", "I")

                            alan_kodu = plaka[:2]
                            geri_kalan = plaka[2:-2]
                            son2rakam = plaka[-2:]
                            print("İşleme verilerine ayrılıyor:")
                            print("Alan kodu:")
                            print(alan_kodu)
                            print("Orta bölüm:")
                            print(geri_kalan)
                            print("Son iki rakam:")
                            print(son2rakam)
                            print("işleniyor...")
                            
                            alan_kodu = str_to_nmb(alan_kodu)

                            if not alan_kodu[0].isalpha():
                                harf_3 = geri_kalan[0]
                                other_cache = geri_kalan[1:]
                                harf_3 = harf_3.replace(" ", "")
                                harf_3 = harf_3.replace("6", "G")
                                harf_3 = harf_3.replace("2", "Z")
                                harf_3 = harf_3.replace("1", "I")
                                harf_3 = harf_3.replace("0", "O")
                                harf_3 = harf_3.replace("5", "S")
                                harf_3 = harf_3.replace("4", "L")
                                harf_3 = harf_3.replace("8", "B")

                                geri_kalan = harf_3 + other_cache
                                geri_kalan = geri_kalan.upper()
                            else:
                                geri_kalan = geri_kalan.upper()
                            
                            son2rakam = str_to_nmb(son2rakam)

                            for _ in range(5):
                                geri_kalan_01 = ""

                                for i in range(len(geri_kalan)):
                                    harf = geri_kalan[i]

                                    if harf == "O":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakterO = geri_kalan[i - 1]
                                            sag_karakterO = geri_kalan[i + 1]

                                            if not (sol_karakterO.isalpha() and sag_karakterO.isalpha()):
                                                geri_kalan_01 += "0"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "0"

                                    elif harf == "I":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakterI = geri_kalan[i - 1]

                                            sag_karakterI = geri_kalan[i + 1]

                                            if not (sol_karakterI.isalpha() and sag_karakterI.isalpha()):
                                                geri_kalan_01 += "1"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "1"

                                    elif harf == "0":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter0 = geri_kalan[i - 1]
                                            sag_karakter0 = geri_kalan[i + 1]

                                            if sol_karakter0.isalpha() and sag_karakter0.isalpha():
                                                geri_kalan_01 += "O"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "0"

                                    elif harf == "1":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter1 = geri_kalan[i - 1]
                                            sag_karakter1 = geri_kalan[i + 1]

                                            if sol_karakter1.isalpha() and sag_karakter1.isalpha():
                                                geri_kalan_01 += "I"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "1"

                                    elif harf == "2":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter2 = geri_kalan[i - 1]
                                            sag_karakter2 = geri_kalan[i + 1]

                                            if sol_karakter2.isalpha() and sag_karakter2.isalpha():
                                                geri_kalan_01 += "Z"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "2"

                                    elif harf == "4":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter4 = geri_kalan[i - 1]
                                            sag_karakter4 = geri_kalan[i + 1]

                                            if sol_karakter4.isalpha() and sag_karakter4.isalpha():
                                                geri_kalan_01 += "l"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "4"

                                    elif harf == "5":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter5 = geri_kalan[i - 1]
                                            sag_karakter5 = geri_kalan[i + 1]

                                            if sol_karakter5.isalpha() and sag_karakter5.isalpha():
                                                geri_kalan_01 += "S"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "5"

                                    elif harf == "6":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter6 = geri_kalan[i - 1]
                                            sag_karakter6 = geri_kalan[i + 1]

                                            if sol_karakter6.isalpha() and sag_karakter6.isalpha():
                                                geri_kalan_01 += "G"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "6"

                                    elif harf == "8":
                                        # Solunda ve sağındaki karakterlerin kontrolü
                                        if 0 < i < len(geri_kalan) - 1:
                                            sol_karakter8 = geri_kalan[i - 1]
                                            sag_karakter8 = geri_kalan[i + 1]

                                            if sol_karakter8.isalpha() and sag_karakter8.isalpha():
                                                geri_kalan_01 += "B"
                                            else:
                                                geri_kalan_01 += harf
                                        else:
                                            geri_kalan_01 += "8"

                                    else:
                                        geri_kalan_01 += harf

                                geri_kalan = geri_kalan_01

                            plaka = alan_kodu + geri_kalan + son2rakam
                            print(plaka)
                            formatted_plate = format_string(plaka)
                            formatted_plate = formatted_plate.replace(" ", "")
                            print(formatted_plate)

                            print("İşleme tamam. Doğruluk kontrol ediliyor...")
                            check_plate = kontrol_et(formatted_plate)
                            if check_plate:
                                print("Plaka Kombinasyonu doğru")
                                print("Tahmini plaka:")
                                print(formatted_plate)

                            else:
                                print("doğrulanamadı")
                        else:
                            print("-----------------------algılanamadı alternatif işlemeya atılıyor--------------------------")
                            alternedPlateText = alternfunc(alternPredict)
                else:
                    print("plaka algılanamadı")

                    continue
        else:
            print("araç algılanamadı")
