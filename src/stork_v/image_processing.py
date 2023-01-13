from typing import Sequence, Tuple
import cv2
import pandas as pd
import numpy as np
import zipfile
from tensorflow import keras
import copy

from stork_v.dataclasses.stork_image import *

# Function to create video
def create_video(
    file_path: str,
    images: Sequence[StorkImage],
    num_frames_from_back=None):

    frame = cv2.imread(images[0].path())
    height, width, layers = frame.shape

    video = cv2.VideoWriter(file_path, 0, 1, (width,height))

    if num_frames_from_back is None:
        mod_images = images
    else:
        mod_images = images[len(images) - num_frames_from_back:]

    for image in mod_images:
        video.write(cv2.imread(image.path()))

    cv2.destroyAllWindows()
    video.release()

# Extract Zip file
def extract_zip(file_path: str, target_directory: str) -> None:
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_directory)
        
# #### Functions for Creating Dataframe

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if np.abs(array[idx] - value) < 0.5:
        return idx
    else:
        return -1

def find_nearest_time(image_intervals: Sequence[float], intervals):
    return [find_nearest(image_intervals, interval) for interval in intervals]

def normalize_maternal_age(maternal_age: float):
    return (maternal_age - 24) / (50 - 24)

# Creating Dataframe for Populating Frame-Timepoint annotations and populating it with -1s
def create_dataframe(image_intervals: Sequence[float], subject_name: str, maternal_age: float, focus = 0, ) -> pd.DataFrame:
    intial_stages = -1
    intervals = np.arange(0.0, 150.0, 0.5)
    data_dict = {}
    for interval in intervals:
        data_dict[str(interval)] = intial_stages
    df_stages = pd.DataFrame(data = data_dict, index=[0])

    # Running functions to get intervals
    df_stages.loc[0] = find_nearest_time(image_intervals, intervals)

    # Populating other information required for model
    df_stages['SUBJECT_NO'] = subject_name
    df_stages['VIDEO'] = df_stages['SUBJECT_NO'].astype(str) + "_" + str(focus) + ".avi"
    df_stages['AGE_AT_RET'] = normalize_maternal_age(maternal_age)
    return df_stages


# #### Functions for Feature Extraction
def build_feature_extractor(img_size: int):
    feature_extractor = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3)
    )

    inputs = keras.Input((img_size, img_size, 3))
    preprocessed = keras.applications.vgg16.preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def crop_petri_dish(frame):
    gimg = copy.deepcopy(frame[:, :, 0])
    cimg = copy.deepcopy(frame)
    circles = cv2.HoughCircles(
        gimg,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=70,
        param2=80,
        minRadius=50,
        maxRadius=120)
    if (circles is not None) and (len(circles) > 0):
        df_circle = pd.DataFrame(data=circles[0], columns=['x', 'y', 'radius'])
        df_circle = df_circle.sort_values(by=['radius'], ascending=False)
        circle = df_circle.iloc[0].values

        mask = np.zeros_like(cimg)
        mask = cv2.circle(mask.copy(), (int(circle[0]),int(circle[1])), int(circle[2]), (255,255,255), -1)
        result = cv2.bitwise_and(cimg, mask)
        return result
    else:
        return frame

def load_video(file_path: str, max_frames: int, resize: Tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(file_path)
    frames = []
    width = None
    height = None
    should_resize = None

    try:
        while True:
            ret, frame = cap.read()
            if width is None or height is None:
                width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not ret:
                break
            frame = crop_center_square(frame)

            if should_resize is None:
                should_resize = resize is not None and \
                (width != resize[0] or height != resize[1])
            
            if should_resize:
                frame = cv2.resize(frame, resize)

            frame = frame[:, :, [2, 1, 0]]
            frame = crop_petri_dish(frame)
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)




