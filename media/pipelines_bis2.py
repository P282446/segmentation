import os
import cv2
import numpy as np
from tqdm import tqdm
from inference.face_segmentation import find_landmarks
from media.image_processing import duoImage
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


def body_pixels_generator(video_path, all_landmarks):
    """
    Générateur qui calcule les masques corporels frame par frame,
    par batch, sans jamais rien écrire sur disque.
    """
    video_capture = cv2.VideoCapture(video_path)
    batch_size = int(config["foreground_detection"]["batch_size"])
    i = 0
    current_batch = []

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
                yield mask.opencv_version
                i += 1
            current_batch = []

    # traite le dernier batch incomplet, s'il en reste
    if current_batch:
        masks = find_body_pixels_batch(
            [f for f, _ in current_batch],
            [l for _, l in current_batch])
        for mask in masks:
            yield mask.opencv_version
            i += 1

    video_capture.release()

def draw_landmarks_and_mask_streaming(output_video_path, all_landmarks,
                                       omniscient_mask_gen, shape):
    """
    Consomme le générateur de masques omniscients au fil de l'eau,
    dessine les zones et écrit directement la vidéo finale.
    Aucun fichier intermédiaire n'est créé sur disque.
    """

    n, h, w, fps = shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for i, mask in enumerate(tqdm(omniscient_mask_gen, total=n, desc="Writing final video")):

        if i == 0:
            print("DEBUG - dtype:", mask.dtype)
            print("DEBUG - shape:", mask.shape)
            print("DEBUG - flags:\n", mask.flags)

        # Force le bon type ET la bonne disposition mémoire pour OpenCV
        mask = np.ascontiguousarray(mask, dtype=np.uint8).copy()

        # Sécurité : si le mask est en niveaux de gris (2D), le convertir en couleur
        if mask.ndim == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

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

            # dtype=np.int32 obligatoire pour cv2.fillPoly
            cv2.fillPoly(mask, pts=[np.array(face_points, dtype=np.int32)], color=(100, 255, 255))
            cv2.fillPoly(mask, pts=[np.array(eyes_points, dtype=np.int32)], color=(100, 255, 0))
            cv2.fillPoly(mask, pts=[np.array(mouth_points, dtype=np.int32)], color=(100, 0, 255))

        out.write(mask)

    out.release()

def full_pipeline(video_path, directory, start_stage=1, end_stage=6):
    """
    Pipeline entièrement en flux (streaming) : aucun fichier intermédiaire
    n'est écrit sur disque, seule la vidéo finale est produite.
    Adapté pour respecter un quota disque limité.
    """
    os.makedirs(directory, exist_ok=True)

    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    final_video_path = os.path.join(directory, "seg_" + input_filename + ".wmv")

    n, h, w, fps = get_video_shape(video_path)
    shape = (n, h, w, fps)

    print("Detecting landmarks...")
    all_landmarks = detect_landmarks_pipeline(video_path)

    print("Computing omniscient landmarks...")
    all_landmarks = compute_omniscient_landmarks_pipeline(all_landmarks)

    print("Computing body pixels + omniscient filter + writing final video (streaming)...")
    raw_masks_gen = body_pixels_generator(video_path, all_landmarks)
    omniscient_masks_gen = omniscient_filter_body_pixels(
        raw_masks_gen, n, inertia=int(config['body_detection']['inertia']))

    draw_landmarks_and_mask_streaming(final_video_path, all_landmarks,
                                        omniscient_masks_gen, shape)

    print(f"Terminé : {final_video_path}")
    return final_video_path
