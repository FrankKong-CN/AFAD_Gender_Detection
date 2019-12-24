import imageio
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import time

model = load_model('model_v1.0.0/AFAD_Gender_Detection_2019_12_6.h5')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        # print(roi_color.shape)

        # 获取人脸图像,转为可预测的numpy array
        image = Image.fromarray(roi_color)
        image = image.resize((48, 48))
        # print(image.size)
        image_2 = np.asarray(image, dtype='float32')
        # print(image_2.shape)
        image_2 = image_2 / 255
        image_2 = image_2.reshape(1, 48, 48, 3)

        prediction = model.predict_classes(image_2)

        if prediction == 0:
            # print("男")
            cv2.putText(img, "male", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            # print("女")
            cv2.putText(img, "female", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # print("prediction:", prediction)

        # time.sleep(0.5)

    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
