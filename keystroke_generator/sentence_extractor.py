import csv
import os

from tqdm import tqdm

if __name__ == '__main__':
    sentences = set()
    for path in tqdm(os.listdir("../../Keystrokes/files"), unit=" files"):
        if not path.endswith("keystrokes.txt"):
            continue
        with open(f"../../Keystrokes/files/{path}") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                sentences.add(f"{row["SENTENCE"]}\n")
    with open("../dataset/sentences.txt", "w", encoding="utf-8") as f:
        f.writelines(sentences)
