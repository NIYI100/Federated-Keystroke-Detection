from flask import Flask, request
from keystroke_generator.keystroke_generator import KeystrokeGenerator
from utils import store_new_keystroke_series
from model.run_network import classify_sentence
import model.train as train
import json
import pandas as pd
import torch
import flask

app = Flask(__name__)


@app.route('/training', methods=['POST'])
def training():
    num_epochs = 1
    query_json = request.json
    print(json.dumps(query_json, indent=1))
    file_location = "/ai_model/dataset/train/train.pt"
    len_new_train_set = store_new_keystroke_series(query_json, file_location)
    if (len_new_train_set % 200) == 0:
        print("train with new dataset")
        train.setup()
        for epoch_idx in range(num_epochs):
            loss_epoch = train.train_epoch(epoch_idx)
            train.validate(epoch_idx, loss_epoch)
    resp = flask.Response(response=query_json, status=200)
    return resp


@app.route('/classification', methods=['POST'])
def classification():
    query_json = request.json
    query_json = [query_json[i] for i in query_json]
    print(json.dumps(query_json, indent=1))
    dask_df = pd.DataFrame(query_json)
    # print(dask_df)
    dask_df['SEQUENCE_START_TIME'] = dask_df.groupby(['testSectionId'])['pressTime'].transform('min')
    dask_df['PRESS_TIME_RELATIVE'] = dask_df['pressTime'] - dask_df['SEQUENCE_START_TIME']
    dask_df['data'] = dask_df[['PRESS_TIME_RELATIVE', 'duration', 'jsKeyCode']].values.tolist()
    # print(dask_df["jsKeyCode"])
    dask_df = dask_df.drop(columns=['pressTime', 'duration', 'testSectionId', 'jsKeyCode',
                                    'SEQUENCE_START_TIME', 'PRESS_TIME_RELATIVE'])
    # print(dask_df)
    tensor_input = torch.Tensor(dask_df['data']).to(dtype=torch.float)
    output = classify_sentence(tensor_input)
    print(f"output: {output}")
    res = "Pass" if output == 1.0 else "Fail"
    print(f"result: {res}")
    resp = flask.Response(response=res, status=200)
    return resp


@app.route('/botclassification', methods=['POST'])
def bot_classification():
    print("generate bot data..")
    query = request.data.decode("utf-8")
    print(f"received sentence: {query}")
    gen = KeystrokeGenerator()
    tensor_input = gen.generate_keystroke(query)
    print(f"tensor: {tensor_input}")
    output = classify_sentence(tensor_input)
    print(f"output: {output}")
    res = "Pass" if output == 0.0 else "Fail"
    print(f"result: {res}")
    resp = flask.Response(response=res, status=200)
    return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
