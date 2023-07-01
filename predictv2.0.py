import os
import numpy as np
import time
from ultralytics import YOLO
import cv2
import pytesseract
import string
from unidecode import unidecode
import matplotlib.pyplot as plt
import pyfirmata
import pandas as pd
import easyocr
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("kütüphaneler yüklendi")
nigh_threshold = 50
# board = pyfirmata.Arduino('COM7')
cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
offset = 17  # plakanın sınırlarını sol taraftan daraltmak için kullanılan değişken
text_final = ""
cache = ""
ainput_dir = os.path.join("data")
athresh_path = os.path.join("data", "thresh.jpg")
aocr_path = os.path.join("data", "ocr.jpg")
adata_path = os.path.join("data", "001.jpg")
current_dir = os.getcwd()
input_dir = os.path.join(current_dir, ainput_dir)
thresh_dir = os.path.join(current_dir, athresh_path)
ocr_dir = os.path.join(current_dir, aocr_path)
data_path = os.path.join(current_dir, adata_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
reader = easyocr.Reader(['en'])
print("parametreler tanımlandı")


# Definitions

def process(imgraw):
    Ha, Wa = imgraw.shape[:2]
    img = cv2.resize(imgraw, (Wa*10, Ha*10))
    area_img = img.shape[0]*img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gurultuazalt = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(gurultuazalt, 50, 255, 0)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 5)
    return(erosion, img)
    
def process2(img):
    area_img = img.shape[0]*img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gurultuazalt = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(gurultuazalt, 50, 255, 0)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 3)
    return(erosion, img)


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


# Kamera ile foto çek

while True:
    ret, frame = cap.read()
    cv2.imwrite(data_path, frame)
    img = cv2.imread(data_path)
    #img = cv2.imread('C:/Users/etnae/Downloads/plaka/20230614_202136.jpg')
    #C:\Users\etnae\Downloads\plaka
    if img is None:
        print("Görüntü yüklenemedi")
        continue  # Bir sonraki döngüye geç
    #aşağıdaki işlem gece görüşü 
    # blue_channel = img[:, :, 2]
    # blue_ratio = np.sum(blue_channel) / (720 * 1280)
    # print(blue_ratio)
    # if blue_ratio < nigh_threshold:
    #    nightimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    eğer mavi değeri 40 tan düşük ise gece modunu açabilir diye dursun

    results = model.predict(source=data_path, show=False)
    bboxes = []
    for result in results:
        if len(result.boxes.xywh) > 0:
            bbox = result.boxes.xywh[0]
            xc, yc, w, h = bbox.tolist()
            bbox_confidence = result.boxes.conf[0]
            bboxes.append(bbox)
            print("plaka değişkenleri atandı")
        else:
            print("plaka algılanamadı")
            continue
            
    license_plate = None

    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        offset = 15
        w = w + 0
        h = h + 0
        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)) + offset:int(xc + (w / 2)), :].copy()
        cv2.imwrite(input_dir + "/yolo.jpg", license_plate)
        

    imgraw = cv2.imread(input_dir + "/yolo.jpg")
    

    # Resmi HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(imgraw, cv2.COLOR_BGR2HSV)
    
    # Mavi renk aralığı
    alt_sinir = np.array([90, 50, 50])  # Mavi renk aralığının alt sınırları
    ust_sinir = np.array([130, 255, 255])  # Mavi renk aralığının üst sınırları
    
    # Mavi renk maskesi
    mask = cv2.inRange(hsv, alt_sinir, ust_sinir)
    
    # Mavi alanları siyahla doldur
    imgraw[np.where(mask)] = [0, 0, 0]


 #----------------------------------------------------------------------------

    
    erosion = process(imgraw)[0]
    img = process(imgraw)[1]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
    plt.show()
    
    cropped_contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Kontür bilgilerini içeren bir DataFrame oluşturma
    data = []
    for i, cnt in enumerate(cropped_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h
        data.append([i+1, x, y, w, h, area, aspect_ratio])
    
    
    df = pd.DataFrame(data, columns=['Sıra No', 'x', 'y', 'width', 'height', 'area', 'aspect_ratio'])
    max_area_row = df[df['area'] == df['area'].max()]
    areaNmb = max_area_row['Sıra No'].values[0]
    areax = max_area_row['x'].values[0]
    areay = max_area_row['y'].values[0]
    areawidth = max_area_row['width'].values[0]
    areaheight = max_area_row['height'].values[0]
    
    print("Sıra No:", areaNmb)
    print("x:", areax)
    print("y:", areay)
    print("width:", areawidth)
    print("height:", areaheight)
    # DataFrame'i ekrana yazdırma
    print(df)

    license_plate_cropped = img[areay:areay+areaheight, areax:areax+areawidth].copy()
    # Kontürleri dikdörtgen içine alma ve sıra numarasını yazdırma
    for i, cnt in enumerate(cropped_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv2.putText(img, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    plt.imshow(cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2RGB))
    plt.show()

    cropped_gray = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
    cropped_thresh = cv2.threshold(cropped_gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

    #erosion2 = process2(license_plate_cropped)[0]
    #img2 = process2(license_plate_cropped)[1]
    
    letters, _ = cv2.findContours(cropped_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letterData = []
    for i, cnt in enumerate(letters):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h
        if (0.20 <= aspect_ratio <= 0.90) and (area>1000
                                              ):
            letterData.append([x, y, w, h])
    for i, cnt in enumerate(letters):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h
        #cv2.rectangle(license_plate_cropped, (x, y), (x+w, y+h), (0, 255, 0), 5)
        #cv2.putText(license_plate_cropped, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #plt.imshow(cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2RGB))
    #plt.show()

    letterData = sorted(letterData, key=lambda box: box[0])
    print(letterData)
    
    letters = []
    for i, box in enumerate(letterData):
        x, y, w, h = box
        
        gray_letter = cv2.cvtColor(license_plate_cropped[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

        noise_reduce_letter = cv2.bilateralFilter(gray_letter, 9, 75, 75)

        ret, thresholed_letter = cv2.threshold(noise_reduce_letter, 50, 255, 0)
        thresholed_letter = cv2.bitwise_not(thresholed_letter)
        thresholed_letter = cv2.cvtColor(thresholed_letter, cv2.COLOR_GRAY2BGR)

        bgH, bgW = thresholed_letter.shape[:2]
        bg = np.zeros((int(bgH * 1.5), int(bgW * 1.5), 3), dtype=np.uint8)
        bgY_offset = int((bg.shape[0] - bgH) / 2)
        bgX_offset = int((bg.shape[1] - bgW) / 2)

        bg[bgY_offset:bgY_offset + bgH, bgX_offset:bgX_offset + bgW] = thresholed_letter

        extended_letter=bg

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(extended_letter, kernel, iterations=1)
        #plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
        #plt.show()

        letters.append(erosion)

    if letters and not all(image is None for image in letters):
        max_height = max(image.shape[0] for image in letters)
        total_length = sum(image.shape[1] for image in letters)

        combined_image = np.zeros((max_height, total_length, 3), dtype=np.uint8)

        x_offset = 0
        for image in letters:
                h_letter, w_letter = image.shape[:2]
                combined_image[0:h_letter, x_offset:x_offset + w_letter] = image
                x_offset += w_letter

        

        plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        plt.show()

#===================Okuma bölümü=======================#
        hcom, wcom = combined_image.shape[:2]
        black_rectangle = np.zeros((int(hcom * 1.3), int(wcom * 1.3), 3), dtype=np.uint8)
        ycom_offset = int((black_rectangle.shape[0] - hcom) / 2)
        xcom_offset = int((black_rectangle.shape[1] - wcom) / 2)
        black_rectangle[ycom_offset:ycom_offset + hcom, xcom_offset:xcom_offset + wcom] = combined_image

        
        erosion = cv2.bitwise_not(black_rectangle)

        hh, ww = erosion.shape[:2]
        erosion = cv2.resize(erosion, (ww//10, hh//10))
        
        plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
        plt.show()
    
        #data = pytesseract.image_to_string(black_rectangle, lang='eng', config='--psm 6')
        #data = pytesseract.image_to_data(black_rectangle, output_type='data.frame')
        
        data = reader.readtext(erosion, paragraph = True)

        data = sorted(data, key=lambda x: x[0][0][0])
        sonuclar = [item[1] for item in data]
        print(sonuclar)
        if len(sonuclar) > 0:
            if len(sonuclar) > 1:
                sonuclar = ''.join(sonuclar)
            
            sonuclar = ''.join(sonuclar)
            sonuclar = sonuclar.replace("$", "S")
            sonuclar = sonuclar.translate(str.maketrans('', '', string.punctuation))
            sonuclar = sonuclar.replace(" ", "")
            
            alan_kodu = sonuclar[:2]
            kpv = sonuclar[2:]
    
            alan_kodu = unidecode(alan_kodu)
            alan_kodu = alan_kodu.translate(str.maketrans('', '', string.punctuation))
            alan_kodu = alan_kodu.replace(" ", "")
            alan_kodu = alan_kodu.replace("G", "6")
            alan_kodu = alan_kodu.replace("Z", "2")
            alan_kodu = alan_kodu.replace("z", "2")
            alan_kodu = alan_kodu.replace("I", "1")
            alan_kodu = alan_kodu.replace("ı", "1")
            alan_kodu = alan_kodu.replace("o", "0")
            alan_kodu = alan_kodu.replace("O", "0")
            alan_kodu = alan_kodu.replace("S", "5")
            alan_kodu = alan_kodu.replace("s", "5")
            alan_kodu = alan_kodu.replace("L", "4")
            alan_kodu = alan_kodu.replace("l", "1")
    
    
            plaka = alan_kodu + kpv
    
            print(plaka)
    
            plaka = plaka.upper()
            dezenlenmis = format_string(plaka)
            remove_space2 = dezenlenmis.replace(" ", "")
            dogrulama = kontrol_et(remove_space2)
    
            if dogrulama:
                if remove_space2 == cache:
                    print("Aynı plaka")
                    print(remove_space2)
                else:
                    print("Plaka doğrulandı")
                    text_final = remove_space2
                    print(text_final)
                    if text_final == "16RAM14" or "16SBL55":
                        # board.digital[13].write(1)
                        # time.sleep(1)
                        # board.digital[13].write(0)
                        print("bişiler doğru")
                    else:
                        print("bişiler yanlış")
                        # board.digital[13].write(0)
                    cache = text_final
    
        
