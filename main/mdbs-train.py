import os
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import pickle
from collections import Counter
import sklearn
import copy
from keras import backend as K
from tensorflow.python.keras.layers import Layer
from keras import initializers, regularizers, constraints
from sklearn.linear_model import LogisticRegression
import random
import scipy



IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

MAX_SEQ_LENGTH = 710

# 1 = EUP vs ANU/CXA
# 3 = EUP vs CXA

split = 1
focus = 0

def NormalizeData(data):
    return (data - 24) / (50 - 24)

def NormalizeBSData(data):
    return data/17

train_df = pd.read_csv("Data/train_intervals.csv")
test_df = pd.read_csv("Data/test_intervals.csv")
train_df_to_filter = copy.deepcopy(train_df)
test_df_to_filter = copy.deepcopy(test_df)
train_df_to_filter['SUBJECT_NO'] = train_df['SUBJECT_NO'].astype(str) + '_' + str(focus) + '.avi'
test_df_to_filter['SUBJECT_NO'] = test_df['SUBJECT_NO'].astype(str) + '_' + str(focus) + '.avi'
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

train_df['AGE_AT_RET'] = train_df['AGE_AT_RET'].apply(NormalizeData)
test_df['AGE_AT_RET'] = test_df['AGE_AT_RET'].apply(NormalizeData)

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

train_videos = os.listdir(f'/Volumes/Elements/CBM/Iman/IVF/train')
test_videos = os.listdir(f'/Volumes/Elements/CBM/Iman/IVF/test')
train_df = train_df[train_df_to_filter['SUBJECT_NO'].isin(train_videos)]
test_df = test_df[test_df_to_filter['SUBJECT_NO'].isin(test_videos)]
train_df = train_df.reset_index()
test_df = test_df.reset_index()

# Run if you want to change split to EUP/ANU vs CxA
if (split == 3):
    train_df['TARGET'] = train_df['PLOIDY_STATUS'].replace('ANU', 'EUP')
    test_df['TARGET'] = test_df['PLOIDY_STATUS'].replace('ANU', 'EUP')
    train_df.sample(5)

train_old_df = train_df
test_old_df = test_df


# #### Remove videos based on criteria

# Hour Criteria
beg_hour = 96.0
end_hour = 112.0

train_df = train_df[train_df[str(beg_hour)] != -1]
train_df = train_df[train_df[str(end_hour)] != -1]
test_df = test_df[test_df[str(end_hour)] != -1]
test_df = test_df[test_df[str(beg_hour)] != -1]


# Grading Criteria
unusable_grades = ['CAV M', 'CM', 'MOR', 'VAC', ]

df_general = pd.read_csv("data/Embryos20190701-ES+Ploidy.csv")

d_train = pd.merge(train_df, df_general[['SUBJECT_NO', 'GRADE']], on='SUBJECT_NO')
d_train.index = train_df.index
d_train = d_train.dropna(subset=['GRADE'])
d_train = d_train[~d_train['GRADE'].isin(unusable_grades)]
train_df = copy.deepcopy(d_train)
train_df = train_df.drop(columns=['GRADE'])

d_test = pd.merge(test_df, df_general[['SUBJECT_NO', 'GRADE']], on='SUBJECT_NO')
d_test.index = test_df.index
d_test = d_test.dropna(subset=['GRADE'])
d_test = d_test[~d_test['GRADE'].isin(unusable_grades)]
test_df = copy.deepcopy(d_test)
test_df = test_df.drop(columns=['GRADE'])


# Only embryos with Blastulation time
df_general = pd.read_csv("data/Embryos20190701-ES+Ploidy.csv")

d_train = pd.merge(train_df, df_general[['SUBJECT_NO', 'tSB']], on='SUBJECT_NO')
d_train.index = train_df.index
d_train = d_train.dropna(subset=['tSB'])
train_df = copy.deepcopy(d_train)

d_test = pd.merge(test_df, df_general[['SUBJECT_NO', 'tSB']], on='SUBJECT_NO')
d_test.index = test_df.index
d_test = d_test.dropna(subset=['tSB'])
test_df = copy.deepcopy(d_test)


# Only embryos with blastocyst scores
df_bs = pd.read_excel('data/Embryos20190701-ES+Blastocyst_Scores.xlsx')
df_bs = df_bs[['SUBJECT_NO','BS', 'Expansion_Score', 'ICM_Score', 'TE_Score']]

d_train = pd.merge(train_df, df_bs, on='SUBJECT_NO')
indices_to_keep = d_train['index'].values
train_df = train_df[train_df['index'].isin(indices_to_keep)]
d_train.index = train_df.index
train_df = copy.deepcopy(d_train)

d_test = pd.merge(test_df, df_bs, on='SUBJECT_NO')
indices_to_keep = d_test['index'].values
test_df = test_df[test_df['index'].isin(indices_to_keep)]
d_test.index = test_df.index
test_df = copy.deepcopy(d_test)


print("Ploidy Distribution of Training Samples:")
train_unique, train_counts = np.unique(train_df['PLOIDY_STATUS'].values, return_counts=True)

print("Ploidy Distribution of Test Samples:")
test_unique, test_counts = np.unique(test_df['PLOIDY_STATUS'].values, return_counts=True)

dict(zip(test_unique, train_counts + test_counts))


train_remove_indices = train_old_df.index.difference(train_df.index)
test_remove_indices = test_old_df.index.difference(test_df.index)


class_vocab, _ = np.unique(train_df['TARGET'].values, return_inverse=True)


age_train_orig_data = (train_df['AGE_AT_RET'].values)
age_test_orig_data = np.squeeze((test_df['AGE_AT_RET'].values))

bs_train_orig_data = (train_df['BS'].values)
bs_test_orig_data = np.squeeze((test_df['BS'].values))

exp_train_orig_data = (train_df['Expansion_Score'].values)
exp_test_orig_data = np.squeeze((test_df['Expansion_Score'].values))

icm_train_orig_data = (train_df['ICM_Score'].values)
icm_test_orig_data = np.squeeze((test_df['ICM_Score'].values))

te_train_orig_data = (train_df['TE_Score'].values)
te_test_orig_data = np.squeeze((test_df['TE_Score'].values))

tB_train_orig_data = (train_df['tSB'].values)
tB_test_orig_data = np.squeeze((test_df['tSB'].values))


# #### Original Data Processing
NUM_FEATURES = 512

filename = f'/Volumes/Elements/CBM/Iman/IVF/PickleFiles/petri_crop_features_cnn_vgg16_full.pckl'
infile = open(filename,'rb')
train_data, train_labels, test_data, test_labels = pickle.load(infile)
infile.close()
train_data = list(train_data)
test_data = list(test_data)

# Run to remove rows based on criteria detailed above
train_features = train_data[0]
train_masks = train_data[1]
test_features = test_data[0]
test_masks = test_data[1]

train_features = np.delete(train_features, train_remove_indices, axis=0)
train_masks = np.delete(train_masks, train_remove_indices, axis=0)

test_features = np.delete(test_features, test_remove_indices, axis=0)
test_masks = np.delete(test_masks, test_remove_indices, axis=0)

train_labels = np.delete(train_labels, train_remove_indices, axis=0)
test_labels = np.delete(test_labels, test_remove_indices, axis=0)

train_data[0] = train_features
train_data[1] = train_masks
test_data[0] = test_features
test_data[1] = test_masks

beginning_index = list(train_df).index(str(beg_hour))
ending_index = list(train_df).index(str(end_hour))

train_df_list = np.array(train_df.values.tolist())
test_df_list = np.array(test_df.values.tolist())

train_interval_list = train_df_list[:, beginning_index:ending_index+1].astype(float)
test_interval_list = test_df_list[:, beginning_index:ending_index+1].astype(float)

new_train_features = np.zeros((train_data[0].shape[0], 1+ending_index-beginning_index, train_data[0].shape[2]))
new_train_masks = np.zeros((train_data[0].shape[0], 1+ending_index-beginning_index))
new_test_features = np.zeros((test_data[0].shape[0], 1+ending_index-beginning_index, test_data[0].shape[2]))
new_test_masks = np.zeros((test_data[0].shape[0], 1+ending_index-beginning_index))

train_features = train_data[0]
train_masks = train_data[1]
test_features = test_data[0]
test_masks = test_data[1]

for i, embryo in enumerate(train_features):
    embryo_intervals = train_interval_list[i]
    embryo_mask = train_masks[i]
    for j in range(0, train_interval_list.shape[1]):
        if int(embryo_intervals[j]) != -1:
            new_train_features[i, j, :] = embryo[int(embryo_intervals[j]), :]
            new_train_masks[i, j] = embryo_mask[int(embryo_intervals[j])]
        else:
            new_train_features[i, j, :] = np.zeros((1, NUM_FEATURES))
            new_train_masks[i, j] = 0
            
for i, embryo in enumerate(test_features):
    embryo_intervals = test_interval_list[i]
    embryo_mask = test_masks[i]
    for j in range(0, test_interval_list.shape[1]):
        if int(embryo_intervals[j]) != -1:
            new_test_features[i, j, :] = embryo[int(embryo_intervals[j]), :]
            new_test_masks[i, j] = embryo_mask[int(embryo_intervals[j])]
        else:
            new_test_features[i, j, :] = np.zeros((1, NUM_FEATURES))
            new_test_masks[i, j] = 0

new_train_masks = np.array(new_train_masks, dtype=bool)
new_test_masks = np.array(new_test_masks, dtype=bool)

print("New Shape of Training Data: " + str(new_train_features.shape))
print("New Shape of Training Masks: " + str(new_train_masks.shape))
print("New Shape of Testing Data: " + str(new_test_features.shape))
print("New Shape of Testing Masks: " + str(new_test_masks.shape))

train_data = [None, None]
test_data = [None, None]

train_data[0] = new_train_features
train_data[1] = new_train_masks
test_data[0] = new_test_features
test_data[1] = new_test_masks

train_labels = copy.deepcopy(train_labels)
test_labels = copy.deepcopy(test_labels)

print("New Shape of Training Data: " + str(train_data[0].shape))
print("New Shape of Training Masks: " + str(train_data[1].shape))
print("New Shape of Testing Data: " + str(test_data[0].shape))
print("New Shape of Testing Masks: " + str(test_data[1].shape))


# #### Sequence Model

# Utility for running experiments.
def pred_class_distribution(y_test, y_pred_coded):
    zero_wrong, zero_right, one_wrong, one_right = [0] * 4
    for i in range(0, len(y_pred_coded)):
        if y_test[i] == 0:
            if y_pred_coded[i] == 0:
                zero_right += 1
            else:
                zero_wrong += 1
        if y_test[i] == 1:
            if y_pred_coded[i] == 1:
                one_right += 1
            else:
                one_wrong += 1
    print("Number of 0 Class Predictions Right: " + str(zero_right))
    print("Number of 0 Class Predictions Wrong: " + str(zero_wrong))
    print("Number of 1 Class Predictions Right: " + str(one_right))
    print("Number of 1 Class Predictions Wrong: " + str(one_wrong))
    
    
def display_metrics(metrics, stand_dev = None):
    if stand_dev is None:
        print("MAE: {}".format(metrics[0][1]))
    else:
        s_mae = str(metrics[0][1]) + " ± " + str(stand_dev[0][1])
        print("MAE: " + s_mae)


# Utility for our sequence model.
def get_sequence_model(MAX_SEQ_LENGTH, age, tB, other_scores):
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    age_input = keras.Input((1,))
    model = bilstm_bsScore_age(frame_features_input, mask_input, age_input)
    return model


def bilstm_bsScore_age(frame_features_input, mask_input, feature):
    lstm_layer = layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False))(
        frame_features_input, mask=mask_input
    )
    feat = keras.layers.Dense(20, activation="relu")(feature)
    x = keras.layers.concatenate([lstm_layer, feat])
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    bs_branch = keras.layers.Dense(1, activation="relu", name='bs_output')(x)
    exp_branch = keras.layers.Dense(1, activation="relu", name='exp_output')(x)
    icm_branch = keras.layers.Dense(1, activation="relu", name='icm_output')(x)
    te_branch = keras.layers.Dense(1, activation="relu", name='te_output')(x)

    rnn_model = keras.Model([frame_features_input, mask_input, feature], [bs_branch, exp_branch, icm_branch, te_branch])
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    
    rnn_model.compile(optimizer=opt,
              loss={'bs_output': 'logcosh', 'exp_output': 'logcosh', 'icm_output': 'logcosh', 'te_output': 'logcosh'},
            )
    
    return rnn_model


def run_cv_experiment(X_train, X_test, X_val, X_train_mask, X_test_mask, X_val_mask,
                      age_train_data, age_test_data, age_val_data, bs_train_data, bs_test_data, bs_val_data,
                      tB_train_data, tB_test_data, tB_val_data,
                      exp_train_data, exp_test_data, exp_val_data,
                      icm_train_data, icm_test_data, icm_val_data,
                      te_train_data, te_test_data, te_val_data,
                      train_labels, test_labels, val_labels, age, tB, other_scores):
    interval = 4
    X_train = X_train[:, 0::interval, :]
    X_test = X_test[:, 0::interval, :]
    X_val = X_val[:, 0::interval, :]
    X_train_mask = X_train_mask[:, 0::interval]
    X_test_mask = X_test_mask[:, 0::interval]
    X_val_mask = X_val_mask[:, 0::interval]
    
    MAX_SEQ_LENGTH = X_train.shape[1]
    print(f"Number of Frames: {MAX_SEQ_LENGTH}")
    
    seq_model = get_sequence_model(MAX_SEQ_LENGTH, age, tB, other_scores)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                       patience=5,
                                       mode='min', verbose=1, restore_best_weights=True)

    if age:
        if tB and other_scores:
            history = seq_model.fit(
                    [X_train, X_train_mask, age_train_data],
                    [bs_train_data, exp_train_data, icm_train_data, te_train_data, tB_train_data], 
                    validation_data=([X_val, X_val_mask, age_val_data], [bs_val_data, exp_val_data, icm_val_data, te_val_data, tB_val_data]),
                    epochs=EPOCHS,
                    callbacks=[es], batch_size=16
                )
            y_pred, _, _, _, _ = seq_model.predict([X_test, X_test_mask, age_test_data])
            
        elif tB:
            history = seq_model.fit(
                    [X_train, X_train_mask, age_train_data],
                    [bs_train_data, tB_train_data], 
                    validation_data=([X_val, X_val_mask, age_val_data], [bs_val_data, tB_val_data]),
                    epochs=EPOCHS,
                    callbacks=[es], batch_size=16
                )
            y_pred, _ = seq_model.predict([X_test, X_test_mask, age_test_data])
            
        elif other_scores:
            history = seq_model.fit(
                    [X_train, X_train_mask, age_train_data],
                    [bs_train_data, exp_train_data, icm_train_data, te_train_data], 
                    validation_data=([X_val, X_val_mask, age_val_data], [bs_val_data, exp_val_data, icm_val_data, te_val_data]),
                    epochs=EPOCHS,
                    callbacks=[es], batch_size=16
                )
            y_pred, _, _, _ = seq_model.predict([X_test, X_test_mask, age_test_data])

        y_pred = y_pred.flatten()
        y_test = bs_test_data
        mae = mean_absolute_error(y_test, y_pred)
    
    else:
        if tB:
            history = seq_model.fit(
                    [X_train, X_train_mask],
                    [bs_train_data, tB_train_data], 
                    validation_data=([X_val, X_val_mask], [bs_val_data, tB_val_data]),
                    epochs=EPOCHS,
                    callbacks=[es], batch_size=16
                )
            y_pred, _ = seq_model.predict([X_test, X_test_mask])
        
        elif other_scores:
            history = seq_model.fit(
                    [X_train, X_train_mask],
                    [bs_train_data, exp_train_data, icm_train_data, te_train_data], 
                    validation_data=([X_val, X_val_mask], [bs_val_data, exp_val_data, icm_val_data, te_val_data]),
                    epochs=EPOCHS,
                    callbacks=[es], batch_size=16
                )
            y_pred, _, _, _ = seq_model.predict([X_test, X_test_mask])

        y_pred = y_pred.flatten()
        y_test = bs_test_data
        mae = mean_absolute_error(y_test, y_pred)

    metrics = []
    ground_truth = []
    predictions = []
    metrics.append(mae)
    
    ground_truth.append(y_test)
    predictions.append(y_pred)

    return history, seq_model, metrics, ground_truth, predictions


def get_cross_validated_scores(X, Y, X_test, y_test, mask, X_test_mask, age_data, age_test_data,
                                                   bs_data, bs_test_data,
                                                   tB_data, tB_test_data,
                                                    exp_data, exp_test_data,
                                                    icm_data, icm_test_data,
                                                    te_data, te_test_data,
                                                    age, tB, other_scores):
    
    cv = sklearn.model_selection.StratifiedKFold(n_splits=4,shuffle=True, random_state=42)
    
    all_metrics = []
    all_predictions = []
    all_test_data = []
    metric = []
    stds = []
    models = []
    count = 1
    
    for train, test in cv.split(X, Y):        
        print(f"Training Model for Split {count}")
        X_train = X[train]
        y_train = Y[train]
        X_train_mask = mask[train]
        age_train_data = age_data[train]
        bs_train_data = bs_data[train]
        tB_train_data = tB_data[train]
        exp_train_data = exp_data[train]
        icm_train_data = icm_data[train]
        te_train_data = te_data[train]
        X_val = X[test]
        y_val = Y[test]
        X_val_mask = mask[test]
        age_val_data = age_data[test]
        bs_val_data = bs_data[test]
        tB_val_data = tB_data[test]
        exp_val_data = exp_data[test]
        icm_val_data = icm_data[test]
        te_val_data = te_data[test]
        
        _, sequence_model, metrics, ground_truth, predictions = run_cv_experiment(X_train, X_test, X_val, 
                                                       X_train_mask, X_test_mask, X_val_mask, 
                                                       age_train_data, age_test_data, age_val_data,
                                                       bs_train_data, bs_test_data, bs_val_data,
                                                    tB_train_data, tB_test_data, tB_val_data,
                                                    exp_train_data, exp_test_data, exp_val_data,
                                                      icm_train_data, icm_test_data, icm_val_data,
                                                      te_train_data, te_test_data, te_val_data,
                                                       y_train, y_test, y_val,
                                                       age, tB, other_scores)
        
        
        models.append(sequence_model)
        count += 1
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_test_data.append(ground_truth)
    
    # Calculating Cross Fold Metrics
    all_metrics = np.array(all_metrics)
    mean_mae = np.round(np.mean(all_metrics[:, 0]), 3)
    metric.append(['mae', mean_mae])
    std_mae = np.round(np.std(all_metrics[:, 0]), 3)
    stds.append(['mae', std_mae])
    
    all_predictions = np.array(all_predictions)
    all_test_data = np.array(all_test_data)
    return all_metrics, metric, stds, models, all_predictions, all_test_data


age_regression = 1
other_scores = 1

if age_regression:
    age_regression_title = '_AgeReg'
else:
    age_regression_title = ''
    
if other_scores:
    other_title = '_OtherScores'
else:
    other_title = ''

hour_range = str(int(beg_hour)) + '_' + str(int(end_hour))
interval_title = '2hours'

EPOCHS = 100
t0 = time.time()
NUM_FEATURES = train_data[0].shape[2]
train_features = train_data[0]
train_masks = train_data[1]
test_features = test_data[0]
test_masks = test_data[1]


all_metrics, metrics, stds, models, all_predictions, all_test_data = get_cross_validated_scores(train_features, train_labels, test_features,
                                                                              test_labels, train_masks, test_masks,
                                                                              age_train_data, age_test_data,
                                                                                bs_train_data, bs_test_data,
                                                                                tB_train_data, tB_test_data,
                                                                                exp_train_data, exp_test_data,
                                                                                icm_train_data, icm_test_data,
                                                                                te_train_data, te_test_data,
                                                                                age_regression, tB, other_scores)
t1 = time.time()
print(f"Time Elpased to Obtain Features: {t1-t0}")
display_metrics(metrics, stand_dev = stds)


save_models = 1

for i, model in enumerate(models):
    model_filepath = f'/Volumes/Elements/CBM/Iman/IVF/Models/Blastocyst Multitask/BILSTM{age_regression_title}{other_title}_{hour_range}_{interval_title}_Aug_CV_{str(i+1)}'
    model.save(model_filepath, save_format="h5")


# Second Step Ploidy Prediction Model
split = 1
age = 1

def display_metrics_lr(metrics, stand_dev = None):
    if stand_dev is None:
        print("Accuracy: {}".format(metrics[0][1]))
        print("AUC: {}".format(metrics[1][1]))
        print("Precision: {}".format(metrics[2][1]))
        print("Recall: {}".format(metrics[3][1]))
        print("F1 Score: {}".format(metrics[4][1]))
    else:
        s_acc = str(metrics[0][1]) + " ± " + str(stand_dev[0][1])
        s_f1 = str(metrics[1][1]) + " ± " + str(stand_dev[1][1])
        s_auc = str(metrics[2][1]) + " ± " + str(stand_dev[2][1])
        s_rec = str(metrics[3][1]) + " ± " + str(stand_dev[3][1])
        s_prec = str(metrics[4][1]) + " ± " + str(stand_dev[4][1])
        print("Accuracy: " + s_acc)
        print("F1 Score: " + s_f1)
        print("AUC: " + s_auc)
        print("Recall: " + s_rec)
        print("Precision: " + s_prec)


## Not using augmented versions for training or testing of ploidy prediction model
train_features = train_data[0]
train_masks = train_data[1]
test_features = test_data[0]
test_masks = test_data[1]
train_labels = train_labels
test_labels = test_labels

if (split == 3):
    cxa_train_old_df = train_df.reset_index(drop=True)
    cxa_test_old_df = test_df.reset_index(drop=True)
    cxa_train_new_df = train_df.reset_index(drop=True)
    cxa_test_new_df = test_df.reset_index(drop=True)
    cxa_train_new_df = cxa_train_new_df[cxa_train_new_df['PLOIDY_STATUS'] != 'ANU']
    cxa_test_new_df = cxa_test_new_df[cxa_test_new_df['PLOIDY_STATUS'] != 'ANU']
    
    cxa_train_remove_indices = cxa_train_old_df.index.difference(cxa_train_new_df.index)
    cxa_test_remove_indices = cxa_test_old_df.index.difference(cxa_test_new_df.index)

    train_features = np.delete(train_features, cxa_train_remove_indices, axis=0)
    train_masks = np.delete(train_masks, cxa_train_remove_indices, axis=0)
    test_features = np.delete(test_features, cxa_test_remove_indices, axis=0)
    test_masks = np.delete(test_masks, cxa_test_remove_indices, axis=0)
    train_labels = np.delete(train_labels, cxa_train_remove_indices, axis=0)
    test_labels = np.delete(test_labels, cxa_test_remove_indices, axis=0)
    
    age_train_ploidy_data = (cxa_train_new_df['AGE_AT_RET'].values)
    age_test_ploidy_data = np.squeeze((cxa_test_new_df['AGE_AT_RET'].values))
    bs_train_ploidy_data = (cxa_train_new_df['BS'].values)
    bs_test_ploidy_data = np.squeeze((cxa_test_new_df['BS'].values)) 
    exp_train_ploidy_data = (cxa_train_new_df['Expansion_Score'].values)
    exp_test_ploidy_data = np.squeeze((cxa_test_new_df['Expansion_Score'].values)) 
    icm_train_ploidy_data = (cxa_train_new_df['ICM_Score'].values)
    icm_test_ploidy_data = np.squeeze((cxa_test_new_df['ICM_Score'].values)) 
    te_train_ploidy_data = (cxa_train_new_df['TE_Score'].values)
    te_test_ploidy_data = np.squeeze((cxa_test_new_df['TE_Score'].values)) 
    
    
if (split == 1):
    age_train_ploidy_data = age_train_orig_data
    age_test_ploidy_data = age_test_orig_data
    bs_train_ploidy_data = bs_train_orig_data
    bs_test_ploidy_data = bs_test_orig_data
    exp_train_ploidy_data = exp_train_orig_data
    exp_test_ploidy_data = exp_test_orig_data
    icm_train_ploidy_data = icm_train_orig_data
    icm_test_ploidy_data = icm_test_orig_data
    te_train_ploidy_data = te_train_orig_data
    te_test_ploidy_data = te_test_orig_data


train_mae_bs = []
train_mae_icm = []
train_mae_exp = []
train_mae_te = []

test_mae_bs = []
test_mae_icm = []
test_mae_exp = []
test_mae_te = []

for i in range(0, 4):
    interval = 4
    X_train = train_features[:, 0::interval, :]
    X_train_mask = train_masks[:, 0::interval]
    X_test = test_features[:, 0::interval, :]
    X_test_mask = test_masks[:, 0::interval]
    seq_model = models[i]
    if age_regression:
        y_train_pred, exp_train_pred, icm_train_pred, te_train_pred = seq_model.predict([X_train, X_train_mask, age_train_ploidy_data])
        y_test_pred, exp_test_pred, icm_test_pred, te_test_pred = seq_model.predict([X_test, X_test_mask, age_test_ploidy_data])
    else:
        y_train_pred, exp_train_pred, icm_train_pred, te_train_pred = seq_model.predict([X_train, X_train_mask])
        y_test_pred, exp_test_pred, icm_test_pred, te_test_pred = seq_model.predict([X_test, X_test_mask])
    train_mae_bs.append(mean_absolute_error(y_train_pred, bs_train_ploidy_data))
    train_mae_exp.append(mean_absolute_error(exp_train_pred, exp_train_ploidy_data))
    train_mae_icm.append(mean_absolute_error(icm_train_pred, icm_train_ploidy_data))
    train_mae_te.append(mean_absolute_error(te_train_pred, te_train_ploidy_data))
    
    test_mae_bs.append(mean_absolute_error(y_test_pred, bs_test_ploidy_data))
    test_mae_exp.append(mean_absolute_error(exp_test_pred, exp_test_ploidy_data))
    test_mae_icm.append(mean_absolute_error(icm_test_pred, icm_test_ploidy_data))
    test_mae_te.append(mean_absolute_error(te_test_pred, te_test_ploidy_data))

print(f"BS Training MAE: {np.mean(train_mae_bs)} ± {np.std(train_mae_bs)}")
print(f"BS Testing MAE: {np.mean(test_mae_bs)} ± {np.std(test_mae_bs)}")
print(f"Exp Training MAE: {np.mean(train_mae_exp)} ± {np.std(train_mae_exp)}")
print(f"Exp Testing MAE: {np.mean(test_mae_exp)} ± {np.std(test_mae_exp)}")
print(f"ICM Training MAE: {np.mean(train_mae_icm)} ± {np.std(train_mae_icm)}")
print(f"ICM Testing MAE: {np.mean(test_mae_icm)} ± {np.std(test_mae_icm)}")
print(f"TE Training MAE: {np.mean(train_mae_te)} ± {np.std(train_mae_te)}")
print(f"TE Testing MAE: {np.mean(test_mae_te)} ± {np.std(test_mae_te)}")



# ### Train and Validate Ploidy Prediction Model
def get_ploidy_data(seq_model, train_features, train_masks, test_features, test_masks,
                   age_train_ploidy_data, age_test_ploidy_data, train_labels, test_labels,
                    age, other_scores, ploidy_other_scores):
    interval = 4
    X_train = train_features[:, 0::interval, :]
    X_train_mask = train_masks[:, 0::interval]
    X_test = test_features[:, 0::interval, :]
    X_test_mask = test_masks[:, 0::interval]
    if age_regression:
        if tB and other_scores:
            y_train_pred, exp_train_pred, icm_train_pred, te_train_pred, _ = seq_model.predict([X_train, X_train_mask, age_train_ploidy_data])
            y_test_pred, exp_test_pred, icm_test_pred, te_test_pred, _ = seq_model.predict([X_test, X_test_mask, age_test_ploidy_data])
        elif other_scores:
            y_train_pred, exp_train_pred, icm_train_pred, te_train_pred = seq_model.predict([X_train, X_train_mask, age_train_ploidy_data])
            y_test_pred, exp_test_pred, icm_test_pred, te_test_pred = seq_model.predict([X_test, X_test_mask, age_test_ploidy_data])
        else:
            y_train_pred, _ = seq_model.predict([X_train, X_train_mask, age_train_ploidy_data])
            y_test_pred, _ = seq_model.predict([X_test, X_test_mask, age_test_ploidy_data])
    else:
        if other_scores:
            y_train_pred, exp_train_pred, icm_train_pred, te_train_pred = seq_model.predict([X_train, X_train_mask])
            y_test_pred, exp_test_pred, icm_test_pred, te_test_pred = seq_model.predict([X_test, X_test_mask])
        else:
            y_train_pred, _ = seq_model.predict([X_train, X_train_mask])
            y_test_pred, _ = seq_model.predict([X_test, X_test_mask])

    train_df2 = pd.DataFrame(y_train_pred, columns=['our_bs'])
    train_df2['Age'] = age_train_ploidy_data
    train_df2['EXP'] = exp_train_pred
    train_df2['ICM'] = icm_train_pred
    train_df2['TE'] = te_train_pred
    train_df2['Target'] = train_labels

    test_df2 = pd.DataFrame(y_test_pred, columns=['our_bs'])
    test_df2['Age'] = age_test_ploidy_data
    test_df2['EXP'] = exp_test_pred
    test_df2['ICM'] = icm_test_pred
    test_df2['TE'] = te_test_pred
    test_df2['Target'] = test_labels
    
    if ploidy_other_scores and age:
        features = ['our_bs', 'Age', 'EXP', 'ICM', 'TE']
    elif ploidy_other_scores:
        features = ['our_bs', 'EXP', 'ICM', 'TE']
    elif age:
        features = ['our_bs', 'Age']
    else:
        features = ['our_bs']
    ploidy_X_train = train_df2[features].values
    ploidy_X_test = test_df2[features].values
    _, ploidy_train_labels = np.unique(train_df2['Target'].values, return_inverse=True)
    _, ploidy_test_labels = np.unique(test_df2['Target'].values, return_inverse=True)
    return ploidy_X_train, ploidy_X_test, ploidy_train_labels, ploidy_test_labels


def run_cv_experiment_lr(X_train, X_test, train_labels, test_labels):
    temp_class_weights = sklearn.utils.compute_class_weight('balanced', classes=np.unique(np.squeeze(train_labels)),
                                                           y=np.squeeze(train_labels))
    class_weight = {0: temp_class_weights[0],
                    1: temp_class_weights[1]}

    param_dist ={'objective':'binary:logistic', 'eval_metric':'logloss'}
    arr = np.array(list(class_weight.values()))
    model = LogisticRegression(random_state=0, class_weight='balanced')
    
    model.fit(X_train, np.squeeze(train_labels))
    y_pred_coded=model.predict(X_test)
    accuracy = accuracy_score(test_labels,y_pred_coded)
    f1 = f1_score(test_labels,y_pred_coded)
    rec = recall_score(test_labels,y_pred_coded)
    prec = precision_score(test_labels,y_pred_coded)
    roc_score = roc_auc_score(test_labels, model.predict_proba(X_test)[:, 1])
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred_coded).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    
    metrics = []
    test_results = []
    predictions = []
    metrics.append(accuracy)
    metrics.append(f1)
    metrics.append(roc_score)
    metrics.append(rec)
    metrics.append(prec)
    
    test_results.append(test_labels)
    predictions.append(model.predict_proba(X_test)[:, 1])

    return model, metrics, test_results, predictions


def get_cross_validated_scores_lr(X, train_labels, X_test, test_labels, X_train_mask, X_test_mask,
                                  age_train_ploidy_data, age_test_ploidy_data, regression_models,
                                  age, other_scores, ploidy_other_scores):
    cv = sklearn.model_selection.StratifiedKFold(n_splits=4,shuffle=True, random_state=42)
    
    all_metrics = []
    all_predictions = []
    all_test_data = []
    metric = []
    stds = []
    models = []
    count = 1
    
    for i in range(0, 4):
        seq_model = regression_models[i]
        ploidy_X_train, ploidy_X_test, ploidy_train_labels, ploidy_test_labels = get_ploidy_data(seq_model,
                                                X, X_train_mask, X_test, X_test_mask,
                                               age_train_ploidy_data, age_test_ploidy_data, train_labels,
                                                test_labels, age, other_scores, ploidy_other_scores)
           
        print(f"Training Model for Split {count}")
        
        model, metrics, test_results, predictions = run_cv_experiment_lr(ploidy_X_train, ploidy_X_test,
                                                                         ploidy_train_labels, ploidy_test_labels)
        
        models.append(model)
        count += 1
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_test_data.append(test_results)
    
    # Calculating Cross Fold Metrics
    all_metrics = np.array(all_metrics)
    np.mean(all_metrics[:, 0])
    mean_acc = np.round(np.mean(all_metrics[:, 0]), 3)
    mean_f1 = np.round(np.mean(all_metrics[:, 1]), 3)
    mean_auc = np.round(np.mean(all_metrics[:, 2]), 3)
    mean_rec = np.round(np.mean(all_metrics[:, 3]), 3)
    mean_prec = np.round(np.mean(all_metrics[:, 4]), 3)
    metric.append(['accuracy', mean_acc]), metric.append(['f1', mean_f1])
    metric.append(['roc', mean_auc]), metric.append(['rec', mean_rec])
    metric.append(['prec', mean_prec])
    std_acc = np.round(np.std(all_metrics[:, 0]), 3)
    std_f1 = np.round(np.std(all_metrics[:, 1]), 3)
    std_auc = np.round(np.std(all_metrics[:, 2]), 3)
    std_rec = np.round(np.std(all_metrics[:, 3]), 3)
    std_prec = np.round(np.std(all_metrics[:, 4]), 3)
    stds.append(['accuracy', std_acc]), stds.append(['f1', std_f1])
    stds.append(['roc', std_auc]), stds.append(['rec', std_rec])
    stds.append(['prec', std_prec])
    
    all_predictions = np.array(all_predictions)
    all_test_data = np.array(all_test_data)
    all_preds = np.concatenate(all_predictions.squeeze())
    all_tests = np.concatenate(all_test_data.squeeze())
    fpr, tpr, t = sklearn.metrics.roc_curve(all_tests, all_preds)
    roc_info = [fpr, tpr, t]
    return all_metrics, metric, stds, roc_info, models


ploidy_other_scores = 0

ploidy_all_metrics, ploidy_metrics, ploidy_stds, _, ploidy_models = get_cross_validated_scores_lr(train_features, train_labels,
                                                        test_features, test_labels, train_masks, test_masks,
                                                      age_train_ploidy_data, age_test_ploidy_data, models, age,
                                                             other_scores, ploidy_other_scores)
display_metrics_lr(ploidy_metrics, stand_dev = ploidy_stds)


save_models = 1

if split == 3:
    split_title = 'EUP_CxA'
if split == 1:
    split_title = 'EUP_ANU'
    
if other_scores:
    other_title = '_OtherScores'
else:
    other_title = ''

if age:
    age_title = '_Age'
else:
    age_title = ''
    
if age_regression:
    age_regression_title = '_AgeReg'
else:
    age_regression_title = ''

for i, model in enumerate(ploidy_models):
    model_filepath = f'/Volumes/Elements/CBM/Iman/IVF/Models/Blastocyst Multitask/{split_title}{age_title}{age_regression_title}{other_title}_CV_{str(i+1)}' 
    pickle.dump(model, open(model_filepath, "wb"))

