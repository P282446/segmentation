import cv2
import argparse
import numpy as np
import configparser
from PIL import Image
from datetime import datetime
from inference.face_segmentation import find_landmarks
from media.image_processing import duoImage, print_all_landmarks_on_image, print_all_zones_on_image, resize_image
"""collections = [[
    458, 131, 171, 133, 174, 167, 84, 95, 127, 121, 120, 119, 118, 117, 270,
    353, 347, 348, 349, 350, 356, 324, 311, 397, 404, 362, 401, 360
]]"""

collections = []


def write_config_file(collections, file_path):
    """
    Write a collection of zones defined by bounding landmarks to a config file
    """
    config = configparser.ConfigParser()
    eyes_region = str(collections[0])
    mouth_region = str(collections[1])
    face_region = [
        457, 358, 400, 364, 413, 446, 305, 340, 233, 374, 335, 409, 295, 331,
        317, 318, 291, 319, 315, 90, 67, 89, 88, 102, 70, 179, 106, 144, 13,
        111, 78, 216, 183, 135, 170, 129, 457
    ]

    config['regions'] = {
        'eyes_region': eyes_region,
        'mouth_region': mouth_region,
        'face_region': face_region,
        **{
            'unnamed__region_' + str(i): collections[i]
            for i in range(2, len(collections))
        }
    }
    config['denoising'] = {
        'denoising_filter_degree': 3,
        'denoising_filter_window': 5
    }

    config['interpolation'] = {'max_interpolation_distance': 10}

    config['face_detection'] = {'rotation_step': 16}

    config['body_detection'] = {'inertia': 8}


    config['foreground_detection'] = {'batch_size': 2}

    with open(file_path, 'w') as configfile:
        config.write(configfile)


def mouse_event(event, x, y, flags, param):
    """
    Update the printed image :
    If the user clicked we add a landmark to the current zone
    If the mouse moves we highlight the closest landmark
    """
    img_ = img.copy()
    ldm_array = np.array(landmarks)
    differences = ldm_array - np.array([x, y])
    distances = np.linalg.norm(differences, axis=1)
    best_index = np.argmin(distances)
    if event == cv2.EVENT_LBUTTONDOWN:
        selected.append(best_index)
        cv2.imshow(
            'image',
            print_all_zones_on_image(img_, landmarks,
                                     collections + [selected]).opencv_version)
    if event == cv2.EVENT_MOUSEMOVE:
        o = img_.opencv_version.copy()
        cv2.circle(o, landmarks[best_index], 5, (0, 255, 0), -1)
        img_.opencv_version = o
        cv2.imshow(
            'image',
            print_all_zones_on_image(img_, landmarks,
                                     collections + [selected]).opencv_version)


if __name__ == "__main__":
    """
    Main function :
    You must specify an image showing a face via the --image argument

    Select two zones or more corresponding to a collection of bounding landmarks


    collections variable contained the saved zones.
    selected variable contains the current zone landmarks.

    Key bindings :
    n to confirm the current selected zone and add it to collections
    c to cancel the last selected point
    q to quit (after selecting at least two zones)

    The user must select the eye zone and then nose/mouth zone.
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image', type=str, help='path to image')
    args = parser.parse_args()
    img = duoImage(Image.open(args.image))
    img = resize_image(img, target_size=2000)
    landmarks = np.array(find_landmarks(img, []))
    selected = []

    img = print_all_landmarks_on_image(img, landmarks)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img.opencv_version)
    cv2.resizeWindow('image', 1600, 1000)
    cv2.setMouseCallback('image', mouse_event)

    while True:
        key = cv2.waitKey(1)
        if key == ord('c'):
            selected = selected[0:-1]
            cv2.imshow(
                'image',
                print_all_zones_on_image(img, landmarks, collections +
                                         [selected]).opencv_version)
        if key == ord('n'):
            collections.append(selected)
            selected = []
            cv2.imshow(
                'image',
                print_all_zones_on_image(img, landmarks, collections +
                                         [selected]).opencv_version)

        if key == ord('q'):
            collections.append(selected)
            if len(collections) >= 2:
                file_path = "configurations/config" + datetime.now().strftime(
                    "%Y%m%d%H%M") + ".txt"
                write_config_file(collections, file_path)

                break

        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
