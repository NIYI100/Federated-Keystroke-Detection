import random

import torch
from torch import Tensor

from keystroke_generator.util import vk_dict, shift_chars


class KeystrokeGenerator:
    """
    A keystroke generator.
    """

    def __init__(self, base_hold_latency: int = 100, base_hold_deviation: int = 70, base_press_latency: int = 160,
                 base_press_deviation: int = 120, rng: random.Random = random.Random()):
        """
        Initializes the keystroke generator.
        :param base_hold_latency: the base latency between the press and the release of a key
        :param base_hold_deviation: how much the hold latency can deviate
        :param base_press_latency: the base latency between the presses of two keys
        :param base_press_deviation: how much the press latency can deviate
        :param rng: instance of a random number generator to randomize the keystroke latencies
        """
        self.base_hold_latency = base_hold_latency
        self.base_hold_deviation = base_hold_deviation
        self.base_press_latency = base_press_latency
        self.base_press_deviation = base_press_deviation
        self.rng = rng

    def generate_keystroke(self, string: str) -> Tensor:
        """
        Generates keystrokes for a given string. Note: only works properly with characters present on a standard US keyboard.
        :param string: the input string
        :return: a tensor of shape `(x, 3)`, where `x` is the length of `string` plus any necessary modifier keystrokes
        (shift, ctrl, alt), and the columns are key press timestamp (ms), key hold time (ms) and JavaScript key code
        """
        now = 0
        idx = 0
        shift_idx = -1
        output = []

        def resolve_shift():
            mean = (min(output[idx - 1][0] + output[idx - 1][1] - output[shift_idx][0],
                        now - output[shift_idx][0] - 1) + output[idx - 1][0] - output[shift_idx][0] + 1) // 2
            dev = mean - (output[idx - 1][0] - output[shift_idx][0] + 1)
            output[shift_idx] = [output[shift_idx][0], self.get_random_number(mean, dev), vk_dict["shift"]]

        for c in string:
            hold_latency = self.get_random_number(self.base_hold_latency, self.base_hold_deviation)
            press_latency = self.get_random_number(self.base_press_latency, self.base_press_deviation, "log-normal")
            if c.isupper() or c in shift_chars:
                if shift_idx == -1:
                    shift_idx = idx
                    output.append([now])
                    now += press_latency
                    idx += 1
                    press_latency = self.get_random_number(self.base_press_latency, self.base_press_deviation,
                                                           "log-normal")
            else:
                if shift_idx != -1:
                    resolve_shift()
                    shift_idx = -1

            if (idx > 1 and len(output[idx - 1]) > 1 and vk_dict[c.lower()] == output[idx - 1][2]
                    and now <= output[idx - 1][0] + output[idx - 1][1]):
                now = output[idx - 1][0] + output[idx - 1][1] + self.get_random_number(self.base_press_deviation,
                                                                                       self.base_press_deviation,
                                                                                       "log-normal")
            output.append([now, hold_latency, vk_dict[c.lower()]])
            now += press_latency
            idx += 1

        if shift_idx != -1:
            resolve_shift()
        print(output)
        return Tensor(output).to(dtype=torch.float)

    def get_random_number(self, mean: int, deviation: int, distribution: str = "normal") -> int:
        if distribution == "normal":
                number = int(self.rng.gauss(mean, deviation / 4))
                while number not in range(mean - deviation, mean + deviation):
                    number = int(self.rng.gauss(mean, deviation / 4))
                return number
        elif distribution == "log-normal":
                mean /= 1000
                deviation /= 1000
                number = self.rng.lognormvariate(-32 / (125 * mean), 27 / (500 * deviation))
                while number < mean - deviation or mean + 1 / (3 * deviation ** 0.5) < number:
                    number = self.rng.lognormvariate(-32 / (125 * mean), 27 / (500 * deviation))
                return int(number * 1000)


if __name__ == '__main__':
    kg = KeystrokeGenerator()
    print(kg.generate_keystroke("HEllo @world!"))
