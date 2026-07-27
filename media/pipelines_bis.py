import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from inference.face_segmentation import find_landmarks
from media.image_processing import print_all_zones_on_image, duoImage
from denoising.smoothing import interpolate_close_missing_values, smooth_defined_data_with_savgol
from configuration_loader import config, eyes_region, mouth_region, face_region
from inference.foreground_detection import find_body_pixels_batch, omniscient_filter_body_pixels


def detect_landmarks_pipeline(video_path):
    video_capture = cv2.VideoCapture(video_path)
    all_landmarks = []
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=frame_count, desc="Detecting landmarks") as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_duo = duoImage(frame)
            all_landmarks.append(
                find_landmarks(frame_duo, all_landmarks[-1] if all_landmarks else []))
            pbar.update(1)
    video_capture.release()
    return all_landmarks


def compute_omniscient_landmarks_pipeline(all_landmarks):
    all_interpolated_landmarks = interpolate_close_missing_values(
        all_landmarks,
        max_distance=int(config['interpolation']['max_interpolation_distance']))
    smoothed_landmarks = smooth_defined_data_with_savgol(all_interpolated_landmarks)
    return [[(int(x), int(y)) for x, y in landmarks]
            for landmarks in tqdm(smoothed_landmarks, desc="Omniscient landmarks")]


def get_video_shape(video_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n, h, w, fps


def compute_body_pixels_to_memmap(video_path, output_path, all_landmarks, shape):
    """
    Calcule les masques corporels et les écrit dans un unique fichier memmap,
    au lieu d'un fichier .jpg par frame (évite ~N créations de fichiers sur NFS).
    """
    n, h, w, _ = shape
    mm = np.memmap(output_path, dtype=np.uint8, mode='w+', shape=(n, h, w, 3))

    video_capture = cv2.VideoCapture(video_path)
    batch_size = int(config["foreground_detection"]["batch_size"])
    i = 0
    current_batch = []
    with tqdm(total=n, desc="Computing body pixels") as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            landmarks = all_landmarks[i]
            if landmarks == []:
                for j in range(1, len(all_landmarks)):
                    if i + j < len(all_landmarks) and all_landmarks[i + j]:
                        landmarks = all_landmarks[i + j]
                        break
                    elif i - j >= 0 and all_landmarks[i - j]:
                        landmarks = all_landmarks[i - j]
                        break
            current_batch.append((duoImage(frame), landmarks))
            if len(current_batch) >= batch_size:
                masks = find_body_pixels_batch(
                    [f for f, _ in current_batch],
                    [l for _, l in current_batch])
                for mask in masks:
                    mm[i] = mask.opencv_version
                    i += 1
                current_batch = []
            pbar.update(1)
    video_capture.release()
    mm.flush()
    del mm


def compute_omniscient_body_pixels_to_memmap(input_path, output_path, shape):
    """
    Applique le filtre omniscient directement entre deux fichiers memmap,
    sans jamais passer par des fichiers image individuels.
    """
    n, h, w, _ = shape
    input_mm = np.memmap(input_path, dtype=np.uint8, mode='r', shape=(n, h, w, 3))

    def mask_generator():
        for i in range(n):
            yield input_mm[i]

    omniscient_gen = omniscient_filter_body_pixels(
        mask_generator(), n, inertia=int(config['body_detection']['inertia']))

    output_mm = np.memmap(output_path, dtype=np.uint8, mode='w+', shape=(n, h, w, 3))
    for i, mask in tqdm(enumerate(omniscient_gen), total=n, desc="Omniscient body pixels"):
        output_mm[i] = mask
    output_mm.flush()
    del output_mm
    del input_mm


def draw_landmarks_and_mask_from_memmap(video_path, output_video_path, all_landmarks,
                                          mask_path, shape):
    n, h, w, _ = shape
    mask_mm = np.memmap(mask_path, dtype=np.uint8, mode='r', shape=(n, h, w, 3))

    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for i in tqdm(range(n), desc="Writing final video"):
        mask = mask_mm[i].copy()  # copie car on modifie le contenu
        landmarks = all_landmarks[i] if i < len(all_landmarks) else []
        if landmarks:
            eyes_points = [landmarks[e] for e in eyes_region]
            mouth_points = [landmarks[m] for m in mouth_region]
            face_points = [landmarks[f] for f in face_region]
            eyes_points = eyes_points + [eyes_points[0]]
            face_points = face_points + [face_points[0]]
            mouth_points = mouth_points + [mouth_points[0]]

            face_bottom = np.max([y for x, y in face_points])
            mask[:, :, :2] = 0
            valid_region = (np.arange(mask.shape[0])[:, None] <= face_bottom) & (mask[:, :, 2] > 0)
            mask[valid_region] = [100, 0, 0]

            cv2.fillPoly(mask, pts=[np.array(face_points)], color=(100, 255, 255))
            cv2.fillPoly(mask, pts=[np.array(eyes_points)], color=(100, 255, 0))
            cv2.fillPoly(mask, pts=[np.array(mouth_points)], color=(100, 0, 255))

        out.write(mask)

    out.release()
    del mask_mm


def full_pipeline(video_path, directory, start_stage=1, end_stage=6):
    """
    Pipeline optimisé : évite l'écriture de milliers de petits fichiers.
    - Étape 3 (visualisation zones) supprimée par défaut (non utilisée).
    - Landmarks gardés en mémoire (pas de JSON écrit, sauf si besoin de reprise).
    - Masques corporels stockés dans 2 fichiers memmap uniques au lieu de
      dizaines de milliers de .jpg individuels.
    """
    os.makedirs(directory, exist_ok=True)

    body_pixels_path = os.path.join(directory, "body_pixels.dat")
    omniscient_body_pixels_path = os.path.join(directory, "omniscient_body_pixels.dat")

    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    final_video_path = os.path.join(directory, "seg_" + input_filename + ".wmv")

    n, h, w, fps = get_video_shape(video_path)
    shape = (n, h, w, 3)

    print("Detecting landmarks...")
    all_landmarks = detect_landmarks_pipeline(video_path)

    print("Computing omniscient landmarks...")
    all_landmarks = compute_omniscient_landmarks_pipeline(all_landmarks)

    print("Computing body pixels...")
    compute_body_pixels_to_memmap(video_path, body_pixels_path, all_landmarks, shape)

    print("Computing omniscient body pixels...")
    compute_omniscient_body_pixels_to_memmap(body_pixels_path, omniscient_body_pixels_path, shape)

    print("Writing final video...")
    draw_landmarks_and_mask_from_memmap(video_path, final_video_path, all_landmarks,
                                          omniscient_body_pixels_path, shape)

    print("Cleaning up intermediate files...")
    for path in [body_pixels_path, omniscient_body_pixels_path]:
        if os.path.isfile(path):
            os.remove(path)

    return final_video_path
