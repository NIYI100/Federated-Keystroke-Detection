import csv
import os

import dask.dataframe as dd
from model.dataset import KeystrokeDataset
import pandas as pd
import torch


def agg_list(dfp, cols):
    return dfp.assign(value=dfp[cols].values.tolist()).drop(columns=cols)


class KeystrokeDataReader:
    def __init__(self, path):
        self.directory_path = path
        self.dataset = KeystrokeDataset()

    def save_torch(self, result, iteration):
        dataset = KeystrokeDataset()
        result = result.groupby('SEQUENCE_ID')['data'].apply(list)
        for val in result:
            dataset.append_item(torch.tensor(val), torch.tensor(1))
        torch.save(dataset, './dataset/' + iteration.__str__() + '_keystroke_dataset.pt')

    # reads all data if file is none
    def read_keystroke_data(self, file_name=None):
        if file_name is None: file_name = '_keystrokes.txt'
        keystrokes_files = [self.directory_path + '/' + f for f in os.listdir(self.directory_path) if
                            f.endswith(file_name)]
        result = pd.DataFrame()

        # columns that we want to process
        columns_to_keep = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']

        # separate list of keystroke files into batches of 10000
        batch_size = 10000
        batched_files = [keystrokes_files[i:i + batch_size] for i in range(0, len(keystrokes_files), batch_size)]

        for i, file_list in enumerate(batched_files):
            print(file_list)
            dask_df = pd.concat([pd.read_csv(f,
                                             encoding='latin-1',
                                             sep='\t',
                                             usecols=columns_to_keep,  # todo: rename
                                             on_bad_lines='skip',
                                             quoting=csv.QUOTE_NONE
                                             ) for f in file_list], ignore_index=True)
            dask_df['SEQUENCE_ID'] = dask_df['PARTICIPANT_ID'].astype(str) + '_' + dask_df['TEST_SECTION_ID'].astype(
                str)
            dask_df = dask_df.drop(columns=['PARTICIPANT_ID', 'TEST_SECTION_ID'])
            # dask_df.to_datetime({'SEQUENCE_ID': ['PARTICIPANT_ID', 'TEST_SECTION_ID']})
            # remove rows with missing values
            dask_df = dask_df.dropna()
            # Calculate sequence start time for each group
            dask_df['SEQUENCE_START_TIME'] = dask_df.groupby(['SEQUENCE_ID'])['PRESS_TIME'].transform('min')

            # Calculate press time relative to sequence start time and press duration
            dask_df['PRESS_TIME_RELATIVE'] = dask_df['PRESS_TIME'] - dask_df['SEQUENCE_START_TIME']
            dask_df['PRESS_DURATION'] = dask_df['RELEASE_TIME'] - dask_df['PRESS_TIME']
            dask_df['data'] = dask_df[['PRESS_TIME_RELATIVE', 'PRESS_DURATION', 'KEYCODE']].values.tolist()
            dask_df = dask_df.drop(columns=['PRESS_TIME', 'RELEASE_TIME', 'SEQUENCE_START_TIME','PRESS_TIME_RELATIVE',
                                            'PRESS_DURATION', 'KEYCODE'])

            self.save_torch(dask_df, i)

        print("1")


if __name__ == "__main__":
    data_reader = KeystrokeDataReader('../../Keystrokes/files')
    # data_reader.check_unique_sequence_participant_ids()
    data_reader.read_keystroke_data()  # '100003_keystrokes.txt'

    columns_to_keep = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']
