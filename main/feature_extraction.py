import os
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2 as cv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import pickle
from collections import Counter
from sklearn import preprocessing
import sklearn
import copy
import random



IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
MAX_SEQ_LENGTH = 710

split = 1
focus = 0


def NormalizeData(data):
    return (data - 24) / (50 - 24)


train_df = pd.read_csv("Data/train_intervals.csv")
test_df = pd.read_csv("Data/test_intervals.csv")
train_df['SUBJECT_NO'] = train_df['SUBJECT_NO'].astype(str) + '_' + str(focus) + '.avi'
test_df['SUBJECT_NO'] = test_df['SUBJECT_NO'].astype(str) + '_' + str(focus) + '.avi'

train_df['AGE_AT_RET'] = train_df['AGE_AT_RET'].apply(NormalizeData)
test_df['AGE_AT_RET'] = test_df['AGE_AT_RET'].apply(NormalizeData)

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

test_df.head(5)

train_df.describe()


# #### Only Run When Not All Videos Are Present

train_videos = os.listdir(f'/Volumes/Elements/CBM/Iman/IVF/train_{focus}')
test_videos = os.listdir(f'/Volumes/Elements/CBM/Iman/IVF/test_{focus}')
train_df = train_df[train_df['SUBJECT_NO'].isin(train_videos)]
test_df = test_df[test_df['SUBJECT_NO'].isin(test_videos)]
train_df = train_df.reset_index()
test_df = test_df.reset_index()

class_vocab, _ = np.unique(train_df['TARGET'].values, return_inverse=True)


# #### Pre-trained network to extract meaningful features from the extracted frames
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

feature_extractor = build_feature_extractor()
print("Output Shape from Feature Extractor: " + str(feature_extractor.output_shape))
NUM_FEATURES = feature_extractor.output_shape[1]

# #### Put all the pieces together to create our data processing utility

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def crop_petri_dish(frame):
    gimg = copy.deepcopy(frame[:, :, 0])
    cimg = copy.deepcopy(frame)
    circles = cv.HoughCircles(gimg,cv.HOUGH_GRADIENT,1,100,
                                param1=70,param2=80,minRadius=50,maxRadius=120)
    if (circles is not None) and (len(circles) > 0):
        df_circle = pd.DataFrame(data=circles[0], columns=['x', 'y', 'radius'])
        df_circle = df_circle.sort_values(by=['radius'], ascending=False)
        circle = df_circle.iloc[0].values

        mask = np.zeros_like(cimg)
        mask = cv.circle(mask.copy(), (int(circle[0]),int(circle[1])), int(circle[2]), (255,255,255), -1)
        result = cv.bitwise_and(cimg, mask)
        return result
    else:
        return frame

def load_video(path, max_frames=710, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frame = crop_petri_dish(frame)
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["SUBJECT_NO"].values.tolist()
    labels = df["TARGET"].values
    _, labels = np.unique(df['TARGET'].values, return_inverse=True)

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
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

        print("Done with " + str(idx+1) + " out of " + str(len(video_paths)) + " videos", end='\r')
        frame_features[idx,] = temp_frame_featutes.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


t0 = time.time()
train_data, train_labels = prepare_all_videos(train_df, f"/Volumes/Elements/CBM/Iman/IVF/train_{focus}")
test_data, test_labels = prepare_all_videos(test_df, f"/Volumes/Elements/CBM/Iman/IVF/test_{focus}")
t1 = time.time()

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")
print(f"Time Elpased to Obtain Features: {t1-t0}")

filename = f'/Volumes/Elements/CBM/Iman/IVF/PickleFiles/petri_crop_features_cnn_vgg16_full_f{focus}.pckl'

outfile = open(filename,'wb')
pickle.dump((train_data, train_labels, test_data, test_labels),outfile)
outfile.close()

