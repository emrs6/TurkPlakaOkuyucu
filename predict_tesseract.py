#import os
from ultralytics import YOLO
import numpy as np
from io import BytesIO
import cv2
#import easyocr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import string
#import re
from unidecode import unidecode


cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
offset = 10
text_final = ""

input_dir = 'C:/Users/etnae/Downloads/automatic-number-plate-recognition-python-master/data'

data_path = "C:/Users/etnae/Downloads/automatic-number-plate-recognition-python-master/data/001.jpg" 
while True:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()

    
    # Görüntüyü belirtilen yol üzerindeki dosyaya yaz
    cv2.imwrite(data_path, frame)
    
    img = cv2.imread(data_path)

    H, W, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
    
    results = model.predict(source=data_path, show=False)

    
    bboxes = []
    class_ids = []
    scores = []
    
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
        
        xc -= offset/2
        
        w += 2 * offset
        h += 2 * offset

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
        if license_plate is not None:
            

            #img = cv2.rectangle(img,
            #                    (int(xc - (w / 2)), int(yc - (h / 2))),
            #                    (int(xc + (w / 2)), int(yc + (h / 2))),
            #                    (0, 255, 0),
            #                    15)

            license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

            

            _, license_plate_thresh = cv2.threshold(license_plate_gray, 65, 255, cv2.THRESH_BINARY_INV)

            
            #output = reader.readtext(license_plate_thresh)
            cv2.imshow("License Plate", license_plate_thresh)
            text_final = ""

            data = pytesseract.image_to_data(license_plate_thresh, output_type='data.frame')

            filtered_data = data[data['conf'] > 40]

            plate_text = ' '.join(filtered_data['text'].values)

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
                

                    
            dezenlenmis = format_string(remove_space)

            remove_space2 = dezenlenmis.replace(" ", "")
            #dogrulama= validate_data(remove_space2)

            print(plate_text)
            print(remove_space)
            print(dezenlenmis)
            print(remove_space2)
            #print(dogrulama)

        else:
            print("No license plate detected")

    if cv2.waitKey() & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
