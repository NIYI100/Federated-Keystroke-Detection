import os

import dask.dataframe as dd
from dask.delayed import delayed
import pandas as pd
class KeystrokeDataReader:
    def __init__(self, path):
        self.directory_path = path

    # reads all data if file is none
    def read_keystroke_data(self, file=None):
        if file == None: file = '*_keystrokes.txt'
        # columns that we want to process
        columns_to_keep = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']

        # read files in the directory
        dask_df = dd.read_csv(self.directory_path + '/' + file, sep='\t', usecols=columns_to_keep,
                              parse_dates={'SEQUENCE_ID': ['PARTICIPANT_ID', 'TEST_SECTION_ID']})

        # Calculate sequence start time for each group
        dask_df['SEQUENCE_START_TIME'] = dask_df.groupby(['SEQUENCE_ID'])['PRESS_TIME'].transform(
            'min')

        # Calculate press time relative to sequence start time
        dask_df['PRESS_TIME_RELATIVE'] = dask_df['PRESS_TIME'] - dask_df['SEQUENCE_START_TIME']

        # Calculate press duration
        dask_df['PRESS_DURATION'] = dask_df['RELEASE_TIME'] - dask_df['PRESS_TIME']

        # Sort values within each group
        dask_df = dask_df.groupby(['SEQUENCE_ID']).apply(
            lambda x: x.sort_values('PRESS_TIME')).reset_index(drop=True)

        # Compute the result
        result = dask_df.compute()

        return result


if __name__ == "__main__":
    data_reader = KeystrokeDataReader('Development/Keystrokes/files')
    keystroke_data = data_reader.read_keystroke_data(file='110414_keystrokes.txt')