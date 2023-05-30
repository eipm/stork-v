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

# In case the zip file has more than 0-focus images, this function only gets the 0-focus images
## Requires focus annotation (XXXX_0.jpg where 0 is indicative of 0 focus)
def create_image_lists(all_embryo_ims):
    ims_pos_0 = []
    for im_name in all_embryo_ims:
        if "_0.jpg" in im_name:
            ims_pos_0.append(im_name)
    return ims_pos_0
    
df = pd.read_csv('Embryos20190701-ES+Ploidy.csv')
df = df[['SUBJECT_NO', 'PLOIDY_STATUS', 'AGE_AT_RET']]
df['TARGET'] = df['PLOIDY_STATUS']
df['TARGET'] = df['TARGET'].replace('CxA', 'ANU')
root = "Temp Zip Files"
zip_files = os.listdir('Temp Zip Files')

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

def create_stages(root, zip_files):
    embryo_nums = [zip_file[0:len(zip_file)-4] for zip_file in zip_files]
    intial_stages = np.zeros(len(embryo_nums)) - 1
    intervals = np.arange(0.0, 150.0, 0.5)
    data_dict = {}
    for interval in intervals:
        data_dict[str(interval)] = intial_stages
    df_stages = pd.DataFrame(data = data_dict)
    for i in range(0, len(embryo_nums)):
        input_directory = root + "/" + zip_files[i]
        target_directory = root + "/" + embryo_nums[i]
        with zipfile.ZipFile(input_directory,"r") as zip_ref:
            zip_ref.extractall(target_directory)
        image_folder = root + "/" + embryo_nums[i] + "/cleaned_good" + "/" + embryo_nums[i]
        all_embryo_ims = os.listdir(image_folder)
        filtered_ims = create_image_lists(all_embryo_ims)
        df_stages.loc[i] = find_nearest_time(filtered_ims, intervals)        
        folder_to_delete = root + "/" + embryo_nums[i]
        shutil.rmtree(folder_to_delete)
    df_stages = df_stages.astype(int)
    df_stages.insert(0, 'SUBJECT_NO', embryo_nums)
    return df_stages

def split_by_fractions(df:pd.DataFrame, fracs:list, random_state:int=42):
    assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
    remain = df.index.copy().to_frame()
    res = []
    for i in range(len(fracs)):
        fractions_sum=sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain=remain.drop(idxs)
        res.append(idxs)
    return [df.loc[idxs] for idxs in res]
 
#### Generate Annotations
df_stages = create_stages(root, zip_files)
df_stages.to_csv("Intervals.csv")
df_stages['SUBJECT_NO'] = df_stages['SUBJECT_NO'].astype(str)
df['SUBJECT_NO'] = df['SUBJECT_NO'].astype(str)
df_new = pd.merge(df, df_stages, left_on='SUBJECT_NO', right_on='SUBJECT_NO', how='left')
train,test = split_by_fractions(df_new, [0.7,0.3]) # e.g: [test, train]
train.to_csv("train_intervals.csv", index=False)
test.to_csv("test_intervals.csv", index=False)
    
#### Creating video from zip file using functions above
root = "/<zip file location>/"
temp_directory = '/<temporary directory>/'
video_directory = '/<video storage directory>/'
zip_files = os.listdir(root)

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

def create_all_videos(root, zip_files):
    embryo_nums = [zip_file[0:len(zip_file)-4] for zip_file in zip_files]
    for i in range(0, len(embryo_nums)):
        focus = 0
        target_directory = temp_directory + embryo_nums[i]
        input_directory = root + "/" + zip_files[i]
        with zipfile.ZipFile(input_directory,"r") as zip_ref:
            zip_ref.extractall(target_directory)
        image_folder = temp_directory + embryo_nums[i] + "/cleaned_good" + "/" + embryo_nums[i]
        all_embryo_ims = os.listdir(image_folder)
        filtered_ims = create_image_lists(all_embryo_ims)        
        if len(filtered_ims) > 10:
            video_name = video_directory + embryo_nums[i] + "_" + str(focus) + '.avi'
            create_video(image_folder, video_name, filtered_ims, num_frames_from_back=None)

train_df = pd.read_csv("Data/train_intervals.csv")
test_df = pd.read_csv("Data/test_intervals.csv")
train_df['SUBJECT_NO'] = train_df['SUBJECT_NO'].astype(str) + '_0.avi'
test_df['SUBJECT_NO'] = test_df['SUBJECT_NO'].astype(str) + '_0.avi'

train_embryos = (train_df['SUBJECT_NO'].values).astype(str)
test_embryos = (test_df['SUBJECT_NO'].values).astype(str)

# Move created videos to train and test folders
all_videos = os.listdir(video_directory)
for video_name in all_videos:
    if video_name in train_embryos:
        shutil.move(video_directory+video_name, '/Volumes/Elements/CBM/Iman/IVF/train/')
    if video_name in test_embryos:
        shutil.move(video_directory+video_name, '/Volumes/Elements/CBM/Iman/IVF/test/')