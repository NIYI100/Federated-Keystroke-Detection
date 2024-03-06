from flask import Flask, request
from keystroke_generator.keystroke_generator import KeystrokeGenerator
from model.run_network import classify_sentence
import json
import pandas as pd
import torch

app = Flask(__name__)


@app.route('/training', methods=['POST'])
def training():
    query_json = request.json
    print(json.dumps(query_json, indent=1))
    # TODO: train local model
    return "", 200


@app.route('/classification', methods=['POST'])
def classification():
    query_json = request.json
    # print(json.dumps(query_json, indent=1))
    dask_df = pd.DataFrame(query_json)
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
    return res, 200


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
    return res, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
