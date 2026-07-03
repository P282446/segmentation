import cv2
from tqdm import tqdm
from image_processing import duoImage


def webcam_analysis_in_real_time(pipeline):
    """
    Displays in real time the result of the provided pipeline on the webcam
    """
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam resolution: {width}x{height}")
    while True:
        ret, frame = cap.read()
        if ret:
            piped_image = pipeline(duoImage(frame))
            cv2.imshow('Webcam Feed', piped_image.opencv_version)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Frame not captured")
            break

    cap.release()
    cv2.destroyAllWindows()


def treat_video(video_path, output_video_path, pipeline, landmarks=[]):
    """
    Treats the input video with the provided pipeline and saves the output video   
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
            if len(landmarks) > 0:
                annotated_frame = pipeline(duoImage(frame), landmarks[i])
            else:
                annotated_frame = pipeline(duoImage(frame))
            out.write(annotated_frame.opencv_version)
            pbar.update(1)
            i += 1
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
