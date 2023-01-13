import pathlib
import shutil
import uuid
import pandas as pd
import numpy as np
import os
import os.path
import pickle
from tensorflow import keras
from typing import List

from stork_v.image_processing import *
from stork_v.dataclasses.experiment_result import *
from stork_v.dataclasses.stork_data import *
from stork_v.dataclasses.stork_result import *
from stork_v.dataclasses.stork_image import *
from os.path import exists

# In case the zip file has more than 0-focus images, this function only gets the 0-focus images
## Requires focus annotation (XXXX_0.jpg where 0 is indicative of 0 focus)


class Stork:
    IMG_SIZE = 224
    MINIMUM_NUMBER_OF_IMAGES = 10
    MAX_SEQ_LENGTH = 710
    NUM_FEATURES = 512
    NUM_SAMPLES = 1 # only doing 1 video at a time right now
    # Hour Criteria (FIXED)
    beg_hour = 96.0
    end_hour = 112.0

    def create_image_list(self, image_paths: Sequence[str]) -> List[StorkImage]:
        stork_images = []
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            filename_no_extension_split = os.path.splitext(filename)[0].split('_')
            if len(filename_no_extension_split) != 5:
                continue
            
            # can raise exception if image filename is not in the correct format    
            hour = float(filename_no_extension_split[2])
            focus =  int(filename_no_extension_split[4]) 

            if focus == 0 and self.beg_hour - 1 < hour < self.end_hour + 1:
                stork_images.append(
                    StorkImage(
                        focus=focus,
                        hour=hour,
                        filename=filename,
                        directory=os.path.dirname(image_path)
                        )
                    )
        return stork_images

    def predict(
        self,
        image_paths: List[str],
        maternal_age: float,
        subject_no: str,
        temp_directory: str,
        focus = 0) -> StorkResult:
        
        stork_images = self.create_image_list(image_paths)
        stork_images.sort(key=lambda x: x.hour)

        if len(stork_images) < self.MINIMUM_NUMBER_OF_IMAGES:
            raise ValueError("Not enough images provided")
        
        video_name = "video_" + str(focus) + '.avi'
        video_path = os.path.join(temp_directory, video_name)
        # Checking if there is at minimum 10 images in the input

        current_file_dir = os.path.dirname(os.path.realpath(__file__))
        temp_video_dir = os.path.join(current_file_dir, '..', 'temp', str(uuid.uuid4()))
        temp_video_path = os.path.join(temp_video_dir, video_name)
        pathlib.Path(os.path.dirname(temp_video_path)).mkdir(parents=True, exist_ok=True)
                
        create_video(temp_video_path, stork_images, num_frames_from_back=None)
        shutil.copy(temp_video_path, video_path)
        shutil.rmtree(temp_video_dir)
        # ### Feature Extraction for Model Input

        # #### Script for Feature Extraction
        # Reads in video and then passes the frames through a pre-trained feature extactor to get features for each frame

        # Creating Dataframe
        image_intervals = list(map(lambda stork_image: stork_image.hour, stork_images))
    
        df_stages = create_dataframe(image_intervals, subject_no, maternal_age)

    #     return self.predict_video(video_path, df_stages)


    # def predict_video(self, video_path: str, df_stages: pd.DataFrame) -> StorkResult:
        if not exists(video_path): 
            raise ValueError("no file can be found")

        frames = load_video(video_path, self.MAX_SEQ_LENGTH, (self.IMG_SIZE, self.IMG_SIZE))
        # Frames shape: (num_frames, height, width, color channels)
        frames = frames[None, ...]
        # Frames shape: (extra_batch_dim, num_frames, height, width, color channels)

        # Initializing pre-trained feature extractor
        feature_extractor = build_feature_extractor(self.IMG_SIZE)

        # `frame_masks` and `frame_features` are what we will feed to our sequence model.
        # `frame_masks` will contain a bunch of booleans denoting if a timestep is
        # masked with padding or not.
        frame_masks = np.zeros(
            shape=(self.NUM_SAMPLES, self.MAX_SEQ_LENGTH),
            dtype="bool")
        frame_features = np.zeros(
            shape=(self.NUM_SAMPLES, self.MAX_SEQ_LENGTH, self.NUM_FEATURES),
            dtype="float32")
        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(self.NUM_SAMPLES, self.MAX_SEQ_LENGTH,),
            dtype="bool")
        temp_frame_featutes = np.zeros(
            shape=(self.NUM_SAMPLES, self.MAX_SEQ_LENGTH, self.NUM_FEATURES),
            dtype="float32")
        
        # Extract features from 
        # the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(self.MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_featutes[i, j, :] = \
                    feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[0,] = temp_frame_featutes.squeeze()
        frame_masks[0,] = temp_frame_mask.squeeze()


        # ### Standardizing Video Features Using Annotation

        # #### Script for Standardization
        # A lot of this I intially coded with the assumption that more than 1 video is being analzyed at the same time
        # The script so far only analyzes 1 video and this part should do the same even though I used for loops to go through (theoretically) multiple embryos

        beginning_index = list(df_stages).index(str(self.beg_hour))
        ending_index = list(df_stages).index(str(self.end_hour))

        df_list = np.array(df_stages.values.tolist())

        interval_list = df_list[:, beginning_index:ending_index+1].astype(float)

        new_features = np.zeros((frame_features.shape[0], 1 + ending_index-beginning_index, frame_features.shape[2]))
        new_masks = np.zeros((frame_features.shape[0], 1 + ending_index-beginning_index))

        for i, embryo in enumerate(frame_features):
            embryo_intervals = interval_list[i]
            embryo_mask = frame_masks[i]
            for j in range(0, interval_list.shape[1]):
                if int(embryo_intervals[j]) != -1:
                    new_features[i, j, :] = embryo[int(embryo_intervals[j]), :]
                    new_masks[i, j] = embryo_mask[int(embryo_intervals[j])]
                else:
                    new_features[i, j, :] = np.zeros((1, self.NUM_FEATURES))
                    new_masks[i, j] = 0

        new_masks = np.array(new_masks, dtype=bool)
        input_data = StorkData(new_features, new_masks)
        
        # #### Script for Running Model on Data
        current_file_dir = os.path.dirname(os.path.realpath(__file__))
        MODELS_FOLDER_PATH = os.path.join(current_file_dir, 'models')
        BS_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH, "BILSTM Model")
        LR_EUP_ANU_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH, "EUP_ANU_LR Model")
        LR_EUP_CXA_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH, "EUP_CxA_LR Model")
        age_data = df_stages['AGE_AT_RET'].values

        lr_eup_anu = self.run_experiment(BS_MODEL_PATH, LR_EUP_ANU_MODEL_PATH, age_data, input_data)
        lr_eup_cxa = self.run_experiment(BS_MODEL_PATH, LR_EUP_CXA_MODEL_PATH, age_data, input_data)
        return StorkResult(lr_eup_anu, lr_eup_cxa)

    def run_experiment(self, model_bs_filepath, model_lr_filepath, age_data, data: StorkData):
        X = np.array(data.features)
        X_mask = np.array(data.masks)

        interval = 4
        X = X[:, 0::interval, :]
        X_mask = X_mask[:, 0::interval]

        print("Shape of Input: " + str(X.shape))

        model_bs = keras.models.load_model(model_bs_filepath)
        model_lr = pickle.load(open(model_lr_filepath, "rb"))

        bs_pred, exp_pred, icm_pred, te_pred = model_bs.predict([X, X_mask, age_data])

        features = ['our_bs', 'Age']
        df_data = pd.DataFrame(bs_pred, columns=['our_bs'])
        df_data['Age'] = age_data
        new_X = df_data[features].values
        y_pred_coded = model_lr.predict(new_X)
        probs = model_lr.predict_proba(new_X)[:, 1]

        return ExperimentResult(
            bs_pred[0][0].item(),
            exp_pred[0][0].item(),
            icm_pred[0][0].item(),
            te_pred[0][0].item(),
            probs[0].item(),
            y_pred_coded[0].item() == True)
        
    def predict_zip_file(
        self,
        zip_file_path: str,
        images_folder_path_in_zip: str = '',
        focus = 0
        ) -> StorkResult:
        
        # Creating video from zip file
        zip_file_name = os.path.basename(zip_file_path)
        zip_file_name_without_extension = os.path.splitext(zip_file_name)[0]

        target_directory = os.path.join(
            os.path.dirname(os.path.realpath(zip_file_path)),
            zip_file_name_without_extension)

        image_paths = self.get_images_from_zip(zip_file_path, target_directory, images_folder_path_in_zip)
        return self.predict(image_paths, maternal_age=30, subject_no=zip_file_name, temp_directory=target_directory, focus=0)
        
    def get_images_from_zip(
        self,
        zip_file_path: str, 
        target_directory: str,
        images_folder_path_in_zip: str = '') -> List[str]:
        
        extract_zip(zip_file_path, target_directory)

        images_directory = os.path.join(target_directory, images_folder_path_in_zip)
        image_paths = [os.path.join(images_directory, image) for image in os.listdir(images_directory)]

        return image_paths
