import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import imutils


def detect_mask_InFrame(frame, face_detect, mask_detect, gender_detect):
    (height, width) = frame.shape[:2]
    gender_list = ['Male', 'Female']
    image_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Performing face detection in the image
    face_detect.setInput(image_blob)
    detections = face_detect.forward()
    location = []
    faces = []
    pred = []
    result = []
    gender_face = []

    # iterate over all the faces
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # Scaling the bounding box as per dimension of our window
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (X_start, Y_start, X_end, Y_end) = box.astype("int")

            # In case face goes out of our frame
            (X_start, Y_start) = (max(X_start, 0), max(Y_start, 0))
            (X_end, Y_end) = (min(X_end, width - 1), min(Y_end, height - 1))

            face = frame[Y_start:Y_end, X_start:X_end]
            # Doing the required pre-processing as per the following research paper
            # link : https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf
            face_Blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=False)
            gender_detect.setInput(face_Blob)
            pred = gender_detect.forward()
            if pred[0][0] > 0.90:
                i = 0
            else:
                i = 1
            gender = gender_list[i]
            gender_face.append(gender)

            # making the input tensor compatible with what our MobilenetV2 fine tuned model expects
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            location.append((X_start, Y_start, X_end, Y_end))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        pred = mask_detect.predict(faces, batch_size=32)

    for i in range(len(faces)):
        result.append((location[i], pred[i], gender_face[i]))
    # return predictions and locations
    return result


def mask_detect():
    # Loading all our models
    prototxtPath = os.path.sep.join(["model_face_detector", "deploy.prototxt.txt"])
    weightsPath = os.path.sep.join(["model_face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    face_detect = cv2.dnn.readNet(prototxtPath, weightsPath)
    prototxtPath = os.path.sep.join(["gender-detector", "deploy_gender.prototxt"])
    weightsPath = os.path.sep.join(["gender-detector", "gender_net.caffemodel"])
    gender_detect = cv2.dnn.readNet(prototxtPath, weightsPath)
    mask_detect = load_model("mask_detector.model")

    # Using Opencv's VideoCapture() method
    cap = cv2.VideoCapture(0)
    # loop over the frames from the input video
    while 1:
        access, frame = cap.read()
        frame = imutils.resize(frame, width=900)
        # Performing mask detection on the fiven frame
        results = detect_mask_InFrame(frame, face_detect, mask_detect, gender_detect)

        for (bounding_box, pred, gender) in results:
            X_start, Y_start, X_end, Y_end = bounding_box
            mask, without_mask = pred

            if mask > 0.6:
                label = "MASK"
            else:
                label = "NO MASK"

            color = (220, 20, 60)

            if label == "MASK":
                color = (25, 25, 112)

            text = label + " : {:.2f}".format(max(mask, without_mask) * 100)
            text2 = "Female"

            cv2.rectangle(frame, (X_start - 4, Y_start), (X_end + 4, Y_start - 40), color, -1)
            cv2.putText(frame, text, (X_start, Y_start - 10), cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 255, 224), 2)
            cv2.rectangle(frame, (X_start, Y_start), (X_end, Y_end), color, 4)
            cv2.rectangle(frame, (X_start - 4, Y_end), (X_end + 4, Y_end + 40), color, -1)
            cv2.putText(frame, text2, (X_start, Y_end + 20), cv2.FONT_HERSHEY_COMPLEX, 0.80, (255, 255, 224), 2)

        cv2.imshow("Face Mask Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()




