from flask import Flask, request
from keystroke_generator.keystroke_generator import KeystrokeGenerator
from utils import store_new_keystroke_series
from model.train import Trainer
from model.classification_network import KeystrokeClassificator
import pickle as pkl
from model.federated_learning import get_parameters, set_parameters
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
    if (len_new_train_set % 2) == 0:
        print("train with new dataset")
        # Load main model
        main_model = KeystrokeClassificator()
        main_model.load_from_path()

        trainer = Trainer(data_folder_path="/ai_model/dataset/train/", prefix="/ai_model/training_output/", model=main_model)
        trainer.train()
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

    # Load main model
    main_model = KeystrokeClassificator()
    main_model.load_from_path()
    output = main_model.classify_sentence(tensor_input)

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
    # Load main model
    main_model = KeystrokeClassificator()
    main_model.load_from_path()
    output = main_model.classify_sentence(tensor_input)

    print(f"output: {output}")
    res = "Pass" if output == 0.0 else "Fail"
    print(f"result: {res}")
    resp = flask.Response(response=res, status=200)
    return resp


@app.route('/getweights', methods=['GET'])
def get_model_weights():
    model_weights = pkl.dumps(get_parameters())
    return flask.Response(model_weights, mimetype="application/octet-stream")


@app.route('/setweights', methods=['POST'])
def set_model_weights():
    model_weights = pkl.loads(request.data)
    set_parameters(model_weights)
    return flask.Response(response="Set weights successfully", status=200)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
