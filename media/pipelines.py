import os
import re
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from inference.face_segmentation import find_landmarks
from media.image_processing import print_all_zones_on_image, duoImage
from denoising.smoothing import  interpolate_close_missing_values, smooth_defined_data_with_savgol
from configuration_loader import config, eyes_region, mouth_region, face_region
from inference.foreground_detection import find_body_pixels_batch, omniscient_filter_body_pixels, omniscient_filter_body_pixels, omniscient_filter_body_pixels_parallelized


def detect_landmarks_pipeline(video_path):
    """
    Detects landmarks for each frame of the input video.
    """
    video_capture = cv2.VideoCapture(video_path)
    all_landmarks = []
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    with tqdm(total=frame_count) as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_duo = duoImage(frame)

            all_landmarks.append(
                find_landmarks(
                    frame_duo,
                    all_landmarks[-1] if len(all_landmarks) > 0 else []))
            pbar.update(1)
    video_capture.release()
    return all_landmarks


def detect_and_save_landmarks_pipeline(video_path, file_path):
    """
    Detects and saves landmarks for each frame of the input video.
    """
    all_landmarks = detect_landmarks_pipeline(video_path)

    with open(file_path, 'w') as outfile:
        json.dump(all_landmarks, outfile)
    return all_landmarks


def compute_omniscient_landmarks_pipeline(all_landmarks):
    """
    Uses interpolation and then Savitzky-Golay filter to find landmarks for each frame that has enough frames with defined landmarks around it
    """
    all_interpolated_landmarks = interpolate_close_missing_values(
        all_landmarks,
        max_distance=int(
            config['interpolation']['max_interpolation_distance']))
    smoothed_landmarks = smooth_defined_data_with_savgol(
        all_interpolated_landmarks)
    return [[(int(landmark[0]), int(landmark[1])) for landmark in landmarks]
            for landmarks in tqdm(smoothed_landmarks)]


def save_omniscient_landmarks_pipeline(all_landmarks, file_path):
    """
    Compute body pixels in omniscient mode for each frame and saves them in the specified directory.
    """
    omniscient_landmarks = compute_omniscient_landmarks_pipeline(all_landmarks)
    with open(file_path, 'w') as outfile:
        json.dump(omniscient_landmarks, outfile)
    return omniscient_landmarks


def draw_landmarks(video_path, output_video_path, all_landmarks):
    """
    Draw detected landmark zones on each frame of the input video, and saves to output video
    """

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    with tqdm(total=frame_count) as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            out.write(
                print_all_zones_on_image(
                    duoImage(frame), all_landmarks[i],
                    [eyes_region, mouth_region, face_region]).opencv_version)
            pbar.update(1)
            i += 1
    video_capture.release()
    out.release()


def image_mask_generator(folder_path):
    """
    Generates images masks from a folder of images.
    This allows to avoid reading all the images at once.
    """
    files = os.listdir(folder_path)
    image_files = [
        f for f in files
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
    ]

    def natural_sort_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]

    image_files.sort(key=natural_sort_key)
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            yield image
        else:
            print(f"Failed to read image: {image_path}")


def compute_and_save_body_pixels(video_path, directory, all_landmarks):
    """
    Computes body pixels for each frame and saves them in the specified directory.
    """
    video_capture = cv2.VideoCapture(video_path)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    current_batch = []
    with tqdm(total=frame_count) as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            landmarks = all_landmarks[i]
            if landmarks == []:
                for j in range(1, len(all_landmarks)):
                    if i + j < len(all_landmarks) and all_landmarks[i + j]:
                        landmarks = all_landmarks[i + j]
                    elif i - j >= 0 and all_landmarks[i - j]:
                        landmarks = all_landmarks[i - j]
            current_batch.append((duoImage(frame),landmarks))
            if len(current_batch) >= int(config["foreground_detection"]["batch_size"]):
                masks = find_body_pixels_batch ([frame for frame, landmarks in current_batch],[landmarks for  frame, landmarks in current_batch])
                for mask in masks:
                    cv2.imwrite(os.path.join(directory,
                                            str(i) + '.jpg'),
                                mask.opencv_version)
                    i += 1
                current_batch = []
            pbar.update (1)
            
    video_capture.release ()


def compute_and_save_omniscient_body_pixels(directory, directory_2):
    """
    Computes the omniscient body pixels for each frame in directory and saves them in directory_2.
    """ 
    image_mask_generator_ = image_mask_generator(directory)
    num_files = len([
        1 for f in os.listdir(directory)
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
    ])
    omniscient_mask_generator = omniscient_filter_body_pixels(
        image_mask_generator_,
        num_files,
        inertia=int(config['body_detection']['inertia']))
    for i, mask in tqdm(enumerate(omniscient_mask_generator), total=num_files):
        cv2.imwrite(os.path.join(directory_2, str(i) + '.jpg'), mask)
        i += 1


def draw_landmarks_and_mask(video_path, output_video_path, all_landmarks,
                            mask_directory):
    """
    Draws landmarks and masks on a video frame based on the provided landmarks and mask directory.
    """

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))
    i = 0
    masks_generator = image_mask_generator(mask_directory)
    for mask in tqdm(masks_generator):
        landmarks = all_landmarks[i]
        if landmarks:
            eyes_points = [landmarks[e] for e in eyes_region]
            mouth_points = [landmarks[m] for m in mouth_region]
            face_points = [landmarks[f] for f in face_region]
            eyes_points = eyes_points + [eyes_points[0]]
            face_points = face_points + [face_points[0]]
            mouth_points = mouth_points + [mouth_points[0]]

            face_bottom = np.max([y for x, y in face_points])
            mask[:, :, :2] = 0
            valid_region = (np.arange(mask.shape[0])[:, None]
                            <= face_bottom) & (mask[:, :, 2] > 0)
            mask[valid_region] = [100, 0, 0]

            cv2.fillPoly(mask,
                         pts=[np.array(face_points)],
                         color=(100, 255, 255))
            cv2.fillPoly(mask,
                         pts=[np.array(eyes_points)],
                         color=(100, 255, 0))
            cv2.fillPoly(mask,
                         pts=[np.array(mouth_points)],
                         color=(100, 0, 255))
            out.write(mask)
        else:
            out.write(mask)
        i += 1
    video_capture.release()
    out.release()


def full_pipeline(video_path, directory, start_stage=1, end_stage=6):
    """
    Goes trough all the specified stages of the video treatment pipeline:
    Stage one : detect landmarks
    Stage two : compute omniscient landmarks
    Stage three : outputs zone visualization
    Stage four : compute body pixels
    Stage five : compute omniscient body pixels
    Stage six : draw detected zones on the video images
    """
    detected_landmarks_file = os.path.join(directory,
                                           "detected_landmarks.json")
    omniscient_landmarks_file = os.path.join(directory,
                                             "omniscient_landmarks.json")
    body_pixels_directory = os.path.join(directory, "body_pixels")
    omniscient_body_pixels_directory = os.path.join(directory,
                                                    "omniscient_body_pixels")
    face_zones_path = os.path.join(directory, "face_zones.wmv")

    # Le nom du fichier final reprend celui de la vidéo d'entrée,
    # mais garde l'extension .wmv définie initialement
    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    final_video_path = os.path.join(directory,"seg_" + input_filename + ".wmv")

    os.makedirs(directory, exist_ok=True)
    os.makedirs(body_pixels_directory, exist_ok=True)
    os.makedirs(omniscient_body_pixels_directory, exist_ok=True)
    stages = range(start_stage, end_stage + 1)
    #stages = [6]
    if start_stage == 2:
        all_landmarks = json.load(open(detected_landmarks_file, 'r'))
    if start_stage >= 3:
        all_landmarks = json.load(open(omniscient_landmarks_file, 'r'))
    if 1 in stages:
        print("Detecting landmarks...")
        all_landmarks = detect_and_save_landmarks_pipeline(
            video_path=video_path, file_path=detected_landmarks_file)
    if 2 in stages:
        print("Computing omniscient landmarks...")
        all_landmarks = save_omniscient_landmarks_pipeline(
            all_landmarks, omniscient_landmarks_file)
    if 3 in stages:
        print("Writing zone visualisation...")
        draw_landmarks(video_path, face_zones_path, all_landmarks)
    if 4 in stages:
        print("Computing body pixels...")
        compute_and_save_body_pixels(video_path, body_pixels_directory,
                                     all_landmarks)
    if 5 in stages:
        print("Computing omniscient body pixels...")
        compute_and_save_omniscient_body_pixels(
            body_pixels_directory, omniscient_body_pixels_directory)
    if 6 in stages:
        print("Writing final video...")
        draw_landmarks_and_mask(video_path, final_video_path, all_landmarks,
                                omniscient_body_pixels_directory)

    # Nettoyage : on ne garde que la vidéo finale
    print("Cleaning up intermediate files...")
    for path in [detected_landmarks_file, omniscient_landmarks_file,
                 face_zones_path]:
        if os.path.isfile(path):
            os.remove(path)

    for path in [body_pixels_directory, omniscient_body_pixels_directory]:
        if os.path.isdir(path):
            shutil.rmtree(path)

   
