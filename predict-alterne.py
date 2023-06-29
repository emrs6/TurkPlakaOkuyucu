import os
import re
import string
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from unidecode import unidecode

print("kütüphaneler yüklendi")
nigh_threshold = 50
cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
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
    ha, wa = imgraw.shape[:2]
    img = cv2.resize(imgraw, (wa * 16, ha * 16))
    return img


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


# =====================================================================#
# Kamera ile foto çek

while True:
    ret, frame = cap.read()
    cv2.imwrite(data_path, frame)
    img = cv2.imread(data_path)
    # img = cv2.imread('C:/Users/etnae/Downloads/plaka/20230614_202136.jpg')
    # C:\Users\etnae\Downloads\plaka
    if img is None:
        print("Görüntü yüklenemedi")
        continue  # Bir sonraki döngüye geç
    # aşağıdaki işlem gece görüşü
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

            license_plate = None

            for bbox_, bbox in enumerate(bboxes):
                xc, yc, w, h = bbox
                w = w + 10
                h = h + 10
                offset = int(w * 0.1)
                license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)) + offset:int(xc + (w / 2)),
                                :].copy()
                cv2.imwrite(input_dir + "/yolo.jpg", license_plate)

            imgraw = cv2.imread(input_dir + "/yolo.jpg")
            if imgraw is None:
                print("Görüntü yüklenemedi")
                continue

            img = process(imgraw)

            license_plate_cropped = img
            Hcache, Wacache = license_plate_cropped.shape[:2]
            license_plate_cropped = cv2.resize(license_plate_cropped, (Wacache // 3, Hcache // 3))

            license_plate_cropped = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            license_plate_cropped = cv2.filter2D(license_plate_cropped, ddepth=-1, kernel=kernel)
            # ret2,license_plate_cropped = cv2.threshold(license_plate_cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()

            # plt.imshow(cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2RGB))
            # plt.show()

            print("okunuyor")

            data = reader.readtext(license_plate_cropped)

            data = sorted(data, key=lambda x: x[0][0][0])
            sonuclar = [item[1] for item in data]

            if len(sonuclar) > 0:
                for a in sonuclar:
                    a.translate(str.maketrans('', '', string.punctuation))
                veriler = []
                if len(sonuclar) < 3:

                    for sonuc in sonuclar:
                        parcalar = sonuc.split()  # Boşluklara göre veriyi parçalara ayır
                        parcalar = re.findall('[A-Za-z]+|\d+', parcalar)
                        veriler.extend(parcalar)
                        sonuclar = veriler

                if len(sonuclar) == 3:
                    veriler = sonuclar

                if not veriler[0].isdigit() and len(veriler) != 0:
                    veriler.remove(veriler[0])

                if len(veriler) == 3:
                    alan_kodu = veriler[0]
                    kpv = veriler[1]
                    son2rakam = veriler[2]

                    # kişiselleştirilmiş plaka verisi
                    print(alan_kodu)
                    print(kpv)
                    print(son2rakam)

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

                    kpv = unidecode(kpv)
                    kpv = kpv.translate(str.maketrans('', '', string.punctuation))
                    kpv = kpv.replace(" ", "")
                    kpv = kpv.replace("6", "G")
                    kpv = kpv.replace("2", "Z")
                    kpv = kpv.replace("1", "I")
                    kpv = kpv.replace("0", "O")
                    kpv = kpv.replace("5", "S")
                    kpv = kpv.replace("4", "L")

                    son2rakam = unidecode(son2rakam)
                    son2rakam = son2rakam.translate(str.maketrans('', '', string.punctuation))
                    son2rakam = son2rakam.replace(" ", "")
                    son2rakam = son2rakam.replace("G", "6")
                    son2rakam = son2rakam.replace("Z", "2")
                    son2rakam = son2rakam.replace("z", "2")
                    son2rakam = son2rakam.replace("I", "1")
                    son2rakam = son2rakam.replace("o", "0")
                    son2rakam = son2rakam.replace("O", "0")
                    son2rakam = son2rakam.replace("S", "5")
                    son2rakam = son2rakam.replace("s", "5")
                    son2rakam = son2rakam.replace("L", "4")
                    son2rakam = son2rakam.replace("l", "1")
                    plaka = alan_kodu + kpv + son2rakam

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
        else:
            print("plaka algılanamadı")
            continue

cap.release()
