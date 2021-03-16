import os
import cv2 as cv
import numpy as np

from os import listdir

if __name__ == '__main__':
    face_cascade_pth = 'D://DeepLearningData/haar_cascade/haarcascade_frontalface_default.xml'
    face_cascade = cv.CascadeClassifier(face_cascade_pth)

    root_samples = './samples'
    pth_list = [os.path.join(root_samples, f) for f in listdir('samples')]

    for pth in pth_list:
        img = cv.imread(pth)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(h, w)
            roi = img[y:y + h, x:x + w]

        cv.imshow('faces', img)
        if cv.waitKey(0) == ord('q'):
            cv.destroyAllWindows()
