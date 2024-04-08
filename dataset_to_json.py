import torch
from model.dataset import KeystrokeDataset
import json


def dataset_to_json(dataset_path, sentences_per_json=None, max_sentences=None):
    """
    :param dataset_path: Dataset path
    :param sentences_per_json: Number of sentences saved in one json file
    :param max_sentences: Total number of sentences converted to json
    :return: None
    """
    dataset: KeystrokeDataset = torch.load(dataset_path)
    if sentences_per_json is None:
        sentences_per_json = len(dataset)

    all_sentences = {}
    json_counter = 0
    for sentence_idx, data in enumerate(dataset):
        if max_sentences and sentence_idx >= max_sentences:
            break
        sentence, label = data
        sentence_json = {}
        for key_idx, keystroke in enumerate(sentence):
            keystroke_json = {
                "PRESS_TIME_RELATIVE": keystroke[0].item(),
                "duration": keystroke[1].item(),
                "jsKeyCode": keystroke[2].item(),
                "testSectionId": sentence_idx
            }
            sentence_json[f"{key_idx}"] = keystroke_json
        all_sentences[f"{sentence_idx}"] = sentence_json

        if (sentence_idx % sentences_per_json) == sentences_per_json-1: # save
            json_string = json.dumps(all_sentences)
            json_path = dataset_path[:-3] + f"_{json_counter}.json"
            with open(json_path, 'w') as file:
                file.write(json_string)
            json_counter += 1



if __name__ == "__main__":
    dataset_path = r'dataset\train\test.pt'
    dataset_to_json(dataset_path, sentences_per_json=1000)
