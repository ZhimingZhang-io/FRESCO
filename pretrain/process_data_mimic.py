import numpy as np
import pandas as pd
import wfdb
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as snss
from pprint import pprint
from tqdm import tqdm
import sys
sys.path.append("../finetune/")
sys.path.append("../utils")
# set your meta path of mimic-ecg
meta_path = '../Dataset/MIMIC-IV-ECG'
report_csv = pd.read_csv(f'{meta_path}/machine_measurements.csv', low_memory=False)
record_csv = pd.read_csv(f'{meta_path}/record_list.csv', low_memory=False)
def process_report(row):
    # Select the relevant columns and filter out NaNs
    report = row[['report_0', 'report_1', 'report_2', 'report_3', 'report_4', 
                  'report_5', 'report_6', 'report_7', 'report_8', 'report_9', 
                  'report_10', 'report_11', 'report_12', 'report_13', 'report_14', 
                  'report_15', 'report_16', 'report_17']].dropna()
    # Concatenate the report
    report = '. '.join(report)
    # Replace and preprocess text
    report = report.replace('EKG', 'ECG').replace('ekg', 'ecg')
    report = report.strip(' ***').strip('*** ').strip('***').strip('=-').strip('=')
    # Convert to lowercase
    report = report.lower()

    # concatenate the report if the report length is not 0
    total_report = ''
    if len(report.split()) != 0:
        total_report = report
        total_report = total_report.replace('\n', ' ')
        total_report = total_report.replace('\r', ' ')
        total_report = total_report.replace('\t', ' ')
        total_report += '.'
    if len(report.split()) == 0:
        total_report = 'empty'
    # Calculate the length of the report in words
    return len(report.split()), total_report

tqdm.pandas()
report_csv['report_length'], report_csv['total_report'] = zip(*report_csv.progress_apply(process_report, axis=1))
# Filter out reports with less than 4 words
report_csv = report_csv[report_csv['report_length'] >= 4]

# you should get 771693 here
print(report_csv.shape)
report_csv.reset_index(drop=True, inplace=True)
record_csv = record_csv[record_csv['study_id'].isin(report_csv['study_id'])]
record_csv.reset_index(drop=True, inplace=True)
# build an empty numpy array to store the data, we use int16 to save the space
temp_npy = np.zeros((len(record_csv), 12, 5000), dtype=np.int16)

for p in tqdm(record_csv['path']):
    # read the data
    ecg_path = os.path.join(meta_path, p)
    record = wfdb.rdsamp(ecg_path)[0]
    record = record.T
    # replace the nan with the neighbor 5 value mean
    # detect nan in each lead
    if np.isnan(record).sum() == 0 and np.isinf(record).sum() == 0:
        # normalize to 0-1
        record = (record - record.min()) / (record.max() - record.min())
        # scale the data
        record *= 1000
        # convert to int16
        record = record.astype(np.int16)
        # store the data
        temp_npy[record_csv[record_csv['path'] == p].index[0]] = record[:, :5000]

    else:
        if np.isinf(record).sum() == 0:
            for i in range(record.shape[0]):
                nan_idx = np.where(np.isnan(record[:, i]))[0]
                for idx in nan_idx:
                    record[idx, i] = np.mean(record[max(0, idx-6):min(idx+6, record.shape[0]), i])
        if np.isnan(record).sum() == 0:
            for i in range(record.shape[0]):
                inf_idx = np.where(np.isinf(record[:, i]))[0]
                for idx in inf_idx:
                    record[idx, i] = np.mean(record[max(0, idx-6):min(idx+6, record.shape[0]), i])

        # normalize to 0-1
        record = (record - record.min()) / (record.max() - record.min())
        # scale the data
        record *= 1000
        # convert to int16
        record = record.astype(np.int16)
        # store the data
        temp_npy[record_csv[record_csv['path'] == p].index[0]] = record[:, :5000]
        
train_npy, val_npy, train_csv, val_csv = train_test_split(temp_npy, report_csv, test_size=0.02, random_state=42)

train_csv.reset_index(drop=True, inplace=True)
val_csv.reset_index(drop=True, inplace=True)

# save to your path
np.save("your_path_train.npy", train_npy)
np.save("your_path_val.npy", val_npy)
train_csv.to_csv("your_path_train.csv", index=False)
val_csv.to_csv("your_path_val.csv", index=False)