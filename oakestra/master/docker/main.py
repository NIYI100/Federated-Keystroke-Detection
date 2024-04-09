import numpy as np
from typing import List
import requests
import pickle
import time


def calculate_averages(parameters: List[List[np.ndarray]]) -> List[np.ndarray]:
    return [np.mean(np.array([model_param[i] for model_param in parameters]), axis=0) for i in
            range(len(parameters[0]))]


def main():
    session = requests.Session()
    session.trust_env = False
    model_weights_list = []
    node_list = ["64.0.0.5", "64.0.0.6", "64.0.0.7"]
    while True:
        try:
            for node in node_list:
                x = session.get(f'http://{node}/getweights')
                curr_model_weights = pickle.loads(x.content)
                print(x)
                model_weights_list.append(curr_model_weights)
            avg_model_weight = pickle.dumps(calculate_averages(model_weights_list))
            model_weights_list = []
            for node in node_list:
                res = session.post(url=f'http://{node}/setweights',
                                    data=avg_model_weight,
                                    headers={'Content-Type': 'application/octet-stream'}
                                    )
                print(res)
            # wait 5 minutes
            time.sleep(60)
        except Exception as e:
            print(e)
            # wait 5 minutes
            time.sleep(60)


if __name__ == '__main__':
    main()