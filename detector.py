import cv2
import numpy as np
import dlib


def draw_landmark(img, faces):
    for i in range(0,5):
        cv2.circle(img, (faces.part(i).x, faces.part(i).y), 1, (255, 0, 0), -1)
        cv2.putText(image, str(i+1), (faces.part(i).x, faces.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return img


args = {
    'image':'faces/SmallFaces.png',
    'model':'models/res10_300x300_ssd_iter_140000.caffemodel',
    'config':'models/deploy.prototxt.txt',
}

net = cv2.dnn.readNetFromCaffe(args["config"], args["model"])

image = cv2.imread(args['image'])
(h, w) = image.shape[:2]
resized = cv2.resize(image, (300, 300))
blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
count = 0
faces = dlib.full_object_detections()
# [(205, 206) (526, 527)]
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence>0.8:
        count += 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        det = dlib.rectangle(startX, startY, endX, endY)
        faces.append(sp(resized[startY:endY,startX:endX], det))
        image = draw_landmark(image, faces[i])
        text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

print(count, len(faces))
cv2.imshow('detections', image)
cv2.waitKey(0)


