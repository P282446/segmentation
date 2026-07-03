import cv2
import math
import mediapipe as mp
from configuration_loader import config
from media.image_processing import new_dimensions, rotate_image, build_base_converting_function, build_rotation_converting_function, correct_uneven_lighting, duoImage, sharpen_image
import torch
from ultralytics import YOLO
from supervision import Detections
import os

os.environ["YOLO_VERBOSE"] = "False"

mpFaceMesh = None
faceMesh = None
yoloModel = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detect_face(image):
    """
    Detects subject's face in an image.

    Given an image, this function uses the YOLOv8 Model to detect
    faces. It returns the face image cropped from the original image, 
    the coordinates of the face rectangle in the original image, and the 
    transformation function that maps coordinates in the face image to coordinates in the original image.
    Each rectangle is represented by its top-left corner (x, y) and its width and
    height.
    """
    global yoloModel
    if yoloModel is None:
        yoloModel = YOLO("models/YOLOv8-face-detection.pt")
        yoloModel.to(device)

    face_margin_coefficient = 1.5
    image_height, image_width = image.size
    image_ = image.copy()
    """faceCascade = cv2.CascadeClassifier(
        "models/haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(image_.opencv_version, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30))"""
    output = yoloModel(image_.pil_version, verbose=False)
    result = Detections.from_ultralytics(output[0])
    if result:
        faces = [[
            result.xyxy[0][0], result.xyxy[0][1],
            result.xyxy[0][2] - result.xyxy[0][0],
            result.xyxy[0][3] - result.xyxy[0][1]
        ]]
    else:
        return None, None, None
    if len(faces) == 0:
        return None, None, None
    (x, y, w, h) = faces[0]
    new_w, new_h = round(w * face_margin_coefficient), round(
        h * face_margin_coefficient)
    new_x, new_y = round(x - w * (face_margin_coefficient - 1) /
                         2), round(y - h * (face_margin_coefficient - 1) / 2)
    new_x = max(new_x, 0)
    new_y = max(new_y, 0)

    new_w = min(new_w, image_width - new_x - 1)
    new_h = min(new_h, image_height - new_y - 1)

    image_.opencv_version = image_.opencv_version[new_y:new_y + new_h,
                                                  new_x:new_x + new_w].copy()

    convert_coordinates = build_base_converting_function(new_x, new_y)

    return image_, [(new_x, new_y, new_w, new_h)], convert_coordinates


def find_landmarks_on_face(image):
    """
    Finds landmarks on a face in an image using mediapipe.

    The input image must be centered on the face.
    """
    global mpFaceMesh
    global faceMesh
    if mpFaceMesh is None:
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    face_rgb = cv2.cvtColor(image.opencv_version, cv2.COLOR_BGR2RGB)
    h, w, _ = face_rgb.shape

    results = faceMesh.process(face_rgb)

    landmarks_list = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for i in range(len(faceLms.landmark)):
                landmarks_list.append(faceLms.landmark.pop())
    return landmarks_list


def find_rotated_face_angle(image, step=16):
    """
    Find the best value for the angle of rotation of an image so that it contains an upward-down face.

    Official documentation of mediapipe suggests a tolerance of 8 degrees for face landmarks detection,
    so the best value of step should be around 16 degrees (assuming that the detect_face function also works face rotated by a few degrees).
    """

    width, height = image.size
    angle_abs = step
    sign = 1
    while angle_abs <= 45:
        angle = angle_abs * sign
        rotation = duoImage(rotate_image(image, angle))
        _, face_coordinates, _ = detect_face(rotation)
        if face_coordinates is not None:
            angle_radian = math.pi * angle / 180
            new_w, new_h = new_dimensions(width, height, angle)
            f1 = build_rotation_converting_function(new_w / 2, new_h / 2,
                                                    width / 2, height / 2,
                                                    -angle_radian)

            face, face_coordinates, f2 = detect_face(rotation)
            f3 = lambda x, y: f1(*f2(x, y))
            return face, angle_radian, f2, rotation, f1
        if sign == 1:
            sign = -1
        else:
            sign = 1
            angle_abs += step
    return None, None, None, None, None


def find_landmarks(image, last_landmarks):
    """
    Finds the face and corresponding landmarks in an image
    Uses last landmarks variable if it is not equal to []

    If we  don't find a face at first, we use the last found landmarks to estimate the position of the face. 
    If the face is still not found , we assume it is rotated.
    If we still don't find it, we run the mediapipe pipeline on the whole image.
    """
    w_orig, h_orig = image.size
    face, _, f = detect_face(image)
    if face == None and len(last_landmarks) > 0:
        last_face_x = min([landmark[0] for landmark in last_landmarks])
        last_face_y = min([landmark[1] for landmark in last_landmarks])
        last_face_w = max([landmark[0]
                           for landmark in last_landmarks]) - last_face_x
        last_face_h = max([landmark[1]
                           for landmark in last_landmarks]) - last_face_y

        new_face_x = last_face_x + last_face_w / 2 - (last_face_w / 2) * 1.5
        new_face_y = last_face_y + last_face_h / 2 - (last_face_h / 2) * 1.5
        new_face_w = last_face_w * 1.5
        new_face_h = last_face_h * 1.5

        new_face_x = int(max(new_face_x, 0))
        new_face_y = int(max(new_face_y, 0))

        new_face_w = int(min(new_face_w, w_orig - new_face_x - 1))
        new_face_h = int(min(new_face_h, h_orig - new_face_y - 1))
        face = duoImage(
            image.opencv_version[new_face_y:new_face_y + new_face_h,
                                 new_face_x:new_face_x + new_face_w])
        f = build_base_converting_function(new_face_x, new_face_y)
        face_upscaled = duoImage(sharpen_image(face))
        w, h = face_upscaled.size
        landmarks = find_landmarks_on_face(face_upscaled)
        if len(landmarks) > 0:

            landmarks = list(map(lambda t: f(t.x * w, t.y * h), landmarks))
            landmarks = list(map(lambda t: (int(t[0]), int(t[1])), landmarks))

            return landmarks
    if face == None:
        face, angle_radian, f2, rotated, f1 = find_rotated_face_angle(
            image, step=int(config['face_detection']['rotation_step']))
        if angle_radian is not None and face.size[0] > 0 and face.size[1] > 0:
            face_upscaled = sharpen_image(face)
            w, h = face_upscaled.size
            landmarks = find_landmarks_on_face(face_upscaled)

            landmarks = list(
                map(lambda t: f1(*f2(t.x * w, t.y * h)), landmarks))
            new_w, new_h = rotated.size
            pos_center_x, pos_center_y = f1(w / 2, h / 2)
            final_landmarks = []
            for x, y in landmarks:
                x_ = int(x - pos_center_x + w / 2 + w_orig / 2 - new_w / 2)
                y_ = int(y - pos_center_y + h / 2 + h_orig / 2 - new_h / 2)
                final_landmarks.append((x_, y_))
            return final_landmarks
    if face == None or face.size[0] == 0 or face.size[1] == 0:
        face = image.copy()
        f = (lambda *x: x)
    face = correct_uneven_lighting(face)
    face_upscaled = sharpen_image(face)
    w, h = face_upscaled.size
    landmarks = find_landmarks_on_face(face_upscaled)
    landmarks = list(map(lambda t: f(t.x * w, t.y * h), landmarks))
    landmarks = list(map(lambda t: (int(t[0]), int(t[1])), landmarks))
    return landmarks
