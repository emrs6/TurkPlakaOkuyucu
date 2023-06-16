import os
from ultralytics import YOLO
import numpy as np
from io import BytesIO
import cv2
import easyocr

cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
offset = 16

input_dir = 'C:/Users/etnae/Downloads/automatic-number-plate-recognition-python-master/data'

data_path = "C:/Users/etnae/Downloads/automatic-number-plate-recognition-python-master/data/001.jpg" 
while True:
    ret, frame = cap.read()

    
    # Görüntüyü belirtilen yol üzerindeki dosyaya yaz
    cv2.imwrite(data_path, frame)
    
    img = cv2.imread(data_path)

    H, W, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
    
    results = model.predict(source=data_path, show=False)

    print(results)
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

    reader = easyocr.Reader(['en'])

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

            output = reader.readtext(license_plate_thresh)
            cv2.imshow("License Plate", license_plate_thresh)
            for out in output:
                text_bbox, text, text_score = out
                if text_score > 0.2:
                    print(text, text_score)
        else:
            print("No license plate detected")

    if cv2.waitKey() & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
