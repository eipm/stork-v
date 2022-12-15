import pandas as pd
import numpy as np
import cv2
import os
import zipfile
import shutil
import pickle
import copy
import random
from pathlib import Path
import warnings

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


# ### Making Video From Zip File

# #### Functions for Making Video

# In case the zip file has more than 0-focus images, this function only gets the 0-focus images
## Requires focus annotation (XXXX_0.jpg where 0 is indicative of 0 focus)
def create_image_lists(all_embryo_ims):
    ims_pos_0 = []
    for im_name in all_embryo_ims:
        if "_0.jpg" in im_name:
            ims_pos_0.append(im_name)
    return ims_pos_0

# Function to create video
## Inputs:
## image_folder - Folder where images are stored --> need this to get the shape of the video (height, weight)
## video_name - name to store video under
## focus_ims - list if images to be stitched together (from one focus)
def create_video(image_folder, video_name, focus_ims, num_frames_from_back=None):
    images = focus_ims
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    if num_frames_from_back is None:
        mod_images = images
    else:
        mod_images = images[len(images) - num_frames_from_back:]

    for image in mod_images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# #### Script to Make a Video From a Zip File

# Creating video from zip file using functions above
zip_file = '78645636.zip'
embryo_num = zip_file[0:len(zip_file)-4]
focus = 0

input_directory = zip_file
target_directory = embryo_num
with zipfile.ZipFile(input_directory,"r") as zip_ref:
    zip_ref.extractall(target_directory)
image_folder = embryo_num + "/cleaned_good" + "/" + embryo_num
all_embryo_ims = os.listdir(image_folder)
filtered_ims = create_image_lists(all_embryo_ims)

# Checking if there is at minimum 10 images in the input
if len(filtered_ims) > 10:
    video_name = embryo_num + "_" + str(focus) + '.avi'
    create_video(image_folder, video_name, filtered_ims, num_frames_from_back=None)


# ### Making Annotation CSV File

# #### Functions for Making Annotation File


def find_beg_hour(ims, stage):
    indices = [idx for idx, s in enumerate(ims) if stage in s]
    if len(indices) > 0:
        first_index = indices[0]
        return int(first_index)
    else:
        return int(-1)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if np.abs(array[idx] - value) < 0.5:
        return idx
    else:
        return -1

def find_nearest_time(ims, intervals):
    frame_nums = []
    image_intervals = []
    for s in ims:
        image_intervals.append(float(s.split('_')[2]))
    for interval in intervals:
        frame_nums.append(find_nearest(image_intervals, interval))
    return frame_nums


# #### Script to Make Annotations

# Creating video from zip file using functions above
zip_file = '78645636.zip'
embryo_num = zip_file[0:len(zip_file)-4]
intervals = np.arange(0.0, 150.0, 0.5)
focus = 0

# Creating Dataframe for Populating Frame-Timepoint annotations and populating it with -1s
intial_stages = -1
intervals = np.arange(0.0, 150.0, 0.5)
data_dict = {}
for interval in intervals:
    data_dict[str(interval)] = intial_stages
df_stages = pd.DataFrame(data = data_dict, index=[0])

input_directory = zip_file
target_directory = embryo_num
with zipfile.ZipFile(input_directory,"r") as zip_ref:
    zip_ref.extractall(target_directory)
image_folder = embryo_num + "/cleaned_good" + "/" + embryo_num
all_embryo_ims = os.listdir(image_folder)
filtered_ims = create_image_lists(all_embryo_ims)

# Running functions to get intervals
df_stages.loc[0] = find_nearest_time(filtered_ims, intervals)

# Populating other information required for model
df_stages['SUBJECT_NO'] = embryo_num
df_stages['VIDEO'] = df_stages['SUBJECT_NO'].astype(str) + "_" + str(focus) + ".avi"
df_stages['AGE_AT_RET'] = 30 # maternal age


# ### Feature Extraction for Model Input
IMG_SIZE = 224

# #### Functions for Feature Extraction


def NormalizeData(data):
    return (data - 24) / (50 - 24)

def build_feature_extractor():
    feature_extractor = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.vgg16.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

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
    circles = cv2.HoughCircles(gimg,cv2.HOUGH_GRADIENT,1,100,
                                param1=70,param2=80,minRadius=50,maxRadius=120)
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

def load_video(path, max_frames=710, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frame = crop_petri_dish(frame)
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


# #### Script for Feature Extraction
# Reads in video and then passes the frames through a pre-trained feature extactor to get features for each frame

MAX_SEQ_LENGTH = 710
NUM_FEATURES = 512
num_samples = 1 # only doing 1 video at a time right now

# Normalizing age
df_stages['AGE_AT_RET'] = df_stages['AGE_AT_RET'].apply(NormalizeData)

# Initializing pre-trained feature extractor
feature_extractor = build_feature_extractor()

root_dir = ""
video_path = df_stages['VIDEO'].values[0]

# `frame_masks` and `frame_features` are what we will feed to our sequence model.
# `frame_masks` will contain a bunch of booleans denoting if a timestep is
# masked with padding or not.
frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
frame_features = np.zeros(
    shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
)

frames = load_video(os.path.join(root_dir, video_path))
# Frames shape: (num_frames, height, width, color channels)
frames = frames[None, ...]
# Frames shape: (extra_batch_dim, num_frames, height, width, color channels)

# Initialize placeholders to store the masks and features of the current video.
temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
temp_frame_featutes = np.zeros(
    shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
)

# Extract features from the frames of the current video.
for i, batch in enumerate(frames):
    video_length = batch.shape[0]
    length = min(MAX_SEQ_LENGTH, video_length)
    for j in range(length):
        temp_frame_featutes[i, j, :] = feature_extractor.predict(
            batch[None, j, :]
        )
    temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

frame_features[0,] = temp_frame_featutes.squeeze()
frame_masks[0,] = temp_frame_mask.squeeze()


# ### Standardizing Video Features Using Annotation

# #### Script for Standardization
# A lot of this I intially coded with the assumption that more than 1 video is being analzyed at the same time
# The script so far only analyzes 1 video and this part should do the same even though I used for loops to go through (theoretically) multiple embryos


# Hour Criteria (FIXED)
beg_hour = 96.0
end_hour = 112.0

data = [frame_features, frame_masks]

beginning_index = list(df_stages).index(str(beg_hour))
ending_index = list(df_stages).index(str(end_hour))

df_list = np.array(df_stages.values.tolist())

interval_list = df_list[:, beginning_index:ending_index+1].astype(float)

new_features = np.zeros((data[0].shape[0], 1+ending_index-beginning_index, data[0].shape[2]))
new_masks = np.zeros((data[0].shape[0], 1+ending_index-beginning_index))

features = data[0]
masks = data[1]

for i, embryo in enumerate(features):
    embryo_intervals = interval_list[i]
    embryo_mask = masks[i]
    for j in range(0, interval_list.shape[1]):
        if int(embryo_intervals[j]) != -1:
            new_features[i, j, :] = embryo[int(embryo_intervals[j]), :]
            new_masks[i, j] = embryo_mask[int(embryo_intervals[j])]
        else:
            new_features[i, j, :] = np.zeros((1, NUM_FEATURES))
            new_masks[i, j] = 0

new_masks = np.array(new_masks, dtype=bool)
data[0] = new_features
data[1] = new_masks


# ### Running Model and Getting Results

# #### Functions for Running Model


def run_experiment(model_bs_filepath, model_lr_filepath, age_data):
    X = np.array(data[0])
    X_mask = np.array(data[1])

    interval = 4
    X = X[:, 0::interval, :]
    X_mask = X_mask[:, 0::interval]

    print("Shape of Input: " + str(X.shape))

    MAX_SEQ_LENGTH = X.shape[1]

    model_bs = keras.models.load_model(model_bs_filepath)
    model_lr = pickle.load(open(model_lr_filepath, "rb"))

    bs_pred, exp_pred, icm_pred, te_pred = model_bs.predict([X, X_mask, age_data])

    features = ['our_bs', 'Age']
    df_data = pd.DataFrame(bs_pred, columns=['our_bs'])
    df_data['Age'] = age_data
    new_X = df_data[features].values
    y_pred_coded = model_lr.predict(new_X)
    probs = model_lr.predict_proba(new_X)[:, 1]

    return bs_pred, exp_pred, icm_pred, te_pred, probs, y_pred_coded


# #### Script for Running Model on Data

model_bs_filepath = "BILSTM Model"
model_lr_filepath_eup_anu = "EUP_ANU_LR Model"
model_lr_filepath_eup_cxa = "EUP_CxA_LR Model"
age_data = df_stages['AGE_AT_RET'].values
bs_pred, exp_pred, icm_pred, te_pred, probs_eup_anu, chr_eup_anu = run_experiment(model_bs_filepath,
                                                                    model_lr_filepath_eup_anu, age_data)
bs_pred, exp_pred, icm_pred, te_pred, probs_eup_cxa, chr_eup_cxa = run_experiment(model_bs_filepath,
                                                                    model_lr_filepath_eup_cxa, age_data)
print(bs_pred, exp_pred, icm_pred, te_pred, probs_eup_anu, chr_eup_anu)
print(bs_pred, exp_pred, icm_pred, te_pred, probs_eup_cxa, chr_eup_cxa)

# bs_pred - predicted BS score from BiLSTM model (ranges from 3 - 14)
# exp_pred - predicted Expansion score from BiLSTM model (ranges from 1 - 4)
# icm_pred - predicted ICM score from BiLSTM model (ranges from 1 - 4)
# te_pred - predicted Trophectoderm score from BiLSTM model (ranges from 1 - 4)

# chr_eup_anu - predicted class for Euploid vs Aneuploid (0 = Aneuploid, 1 = Euploidy)
# chr_eup_cxa - predicted class for Euploid vs Complex Aneuploid (0 = Complex Aneuploid, 1 = Euploidy)
# probs_eup_anu - probability for Euploid vs Aneuploid
# probs_eup_cxa - probability for Euploid vs Complex Aneuploid