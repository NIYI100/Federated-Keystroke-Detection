import torch
from model.dataset import KeystrokeDataset
import pandas as pd


def create_dataset(result):
    dataset = KeystrokeDataset()
    result = result.groupby('testSectionId')['data'].apply(list)
    for val in result:
        dataset.append_item(torch.tensor(val), torch.tensor(1))
    return dataset


def generate_tensor_from_series(series_json):
    dask_df = pd.DataFrame(series_json)
    dask_df['SEQUENCE_START_TIME'] = dask_df.groupby(['testSectionId'])['pressTime'].transform('min')
    dask_df['PRESS_TIME_RELATIVE'] = dask_df['pressTime'] - dask_df['SEQUENCE_START_TIME']
    dask_df['data'] = dask_df[['PRESS_TIME_RELATIVE', 'duration', 'jsKeyCode']].values.tolist()
    dask_df = dask_df.drop(columns=['pressTime', 'duration', 'jsKeyCode',
                                    'SEQUENCE_START_TIME', 'PRESS_TIME_RELATIVE'])

    return dask_df


def store_new_keystroke_series(request_json, file_location):
    try:
        dataset = torch.load(file_location)
    except Exception as e:
        dataset = None
    result = []
    request_list = [request_json[i] for i in request_json]
    for series in request_list:
        series_list = [series[i] for i in series]
        result.append(generate_tensor_from_series(series_list))

    result = pd.concat(result, axis=0)
    result_dataset = create_dataset(result)
    if dataset is not None:
        result_dataset = torch.utils.data.ConcatDataset([dataset, result_dataset])
    torch.save(result_dataset, file_location)
    return len(result_dataset)
