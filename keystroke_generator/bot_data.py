"""
This script generates keystroke datasets. It creates `files` dataset files with `profiles` bot profiles. Each profile is
used to generate keystrokes for 15 sentences. Each sentence has on average 50 keystrokes. That makes ~50 * 15 *
`profiles` * `files` keystrokes. In the current configuration, this script creates ~120,000,000 keystrokes in about 10
minutes.
"""
import random

import torch
from tqdm import tqdm

from keystroke_generator import KeystrokeGenerator
from model.dataset import KeystrokeDataset

if __name__ == '__main__':
    files = 16
    profiles = 10000

    rng = random.Random()
    base_hold_latency = 100
    base_hold_deviation = 70
    base_press_latency = 160
    base_press_deviation = 120
    rng = random.Random()
    prog_bar = tqdm(total=files * profiles, desc="Generating bot data")
    with open("../dataset/sentences.txt") as f:
        sentences = [w.removesuffix("\n") for w in f.readlines()]

    kg = KeystrokeGenerator(rng=rng)

    for i in range(files):
        ds = KeystrokeDataset()
        for j in range(profiles):
            hold_latency = int(rng.gauss(base_hold_latency, base_hold_deviation / 4))
            while hold_latency not in range(base_hold_latency - base_hold_deviation,
                                            base_hold_latency + base_hold_deviation):
                hold_latency = int(rng.gauss(base_hold_latency, base_hold_deviation / 4))

            hold_deviation = int(rng.gauss(hold_latency * 0.75, hold_latency * 0.75 * 0.5 / 4))
            while hold_deviation not in range(int(hold_latency * 0.375), int(hold_latency * 0.875)):
                hold_deviation = int(rng.gauss(hold_latency * 0.75, hold_latency * 0.75 * 0.5 / 4))

            press_latency = rng.lognormvariate(-1.6, 0.45)
            while (press_latency * 1000 < hold_latency * 0.8
                   or press_latency * 1000 < base_press_latency - base_press_deviation
                   or base_press_latency / 1000 + 1 / (3 * (base_press_deviation / 1000) ** 0.5) < press_latency):
                press_latency = rng.lognormvariate(-1.6, 0.45)
            press_latency = int(press_latency * 1000)

            press_deviation = int(rng.gauss(press_latency * 0.75, press_latency * 0.75 * 0.5 / 4))
            while press_deviation not in range(int(press_latency * 0.375), int(press_latency * 0.875)):
                press_deviation = int(rng.gauss(press_latency * 0.75, press_latency * 0.75 * 0.5 / 4))

            kg = KeystrokeGenerator(hold_latency, hold_deviation, press_latency, press_deviation, rng)
            for sentence in rng.choices(sentences, k=15):
                datapoint = kg.generate_keystroke(sentence)
                ds.append_item(datapoint, torch.tensor(0))
            prog_bar.update()
        torch.save(ds, f"../dataset/bot/{i}_bot_keystroke_dataset.pt")
