import random

import torch
from torch import Tensor

from util import vk_dict, shift_chars, ctrl_alt_chars


class KeystrokeGenerator:
    """
    A keystroke generator.
    """

    def __init__(self, base_hold_latency: int = 100, base_hold_deviation: int = 70, base_press_latency: int = 160,
                 base_press_deviation: int = 120, rng: random.Random = random.Random(), distribution: str = "normal"):
        """
        Initializes the keystroke generator.
        :param base_hold_latency: the base latency between the press and the release of a key
        :param base_hold_deviation: how much the hold latency can deviate
        :param base_press_latency: the base latency between the presses of two keys
        :param base_press_deviation: how much the press latency can deviate
        :param rng: instance of a random number generator to randomize the keystroke latencies
        :param distribution: the kind of probability distribution to randomize the keystroke latencies (normal, uniform)
        """
        self.base_hold_latency = base_hold_latency
        self.base_hold_deviation = base_hold_deviation
        self.base_press_latency = base_press_latency
        self.base_press_deviation = base_press_deviation
        self.rng = rng
        self.distribution = distribution

    def generate_keystroke(self, string: str) -> Tensor:
        """
        Generates keystrokes for a given string. Note: only allows characters available on a german keyboard.
        :param string: the input string
        :return: a tensor of shape `(x, 3)`, where `x` is the length of `string` plus any necessary modifier keystrokes
        (shift, ctrl, alt), and the columns are key press timestamp (ms), key hold time (ms) and Windows virtual key
        code
        """
        now = 0
        idx = 0
        shift_idx = -1
        ctrl_alt_idx = -1
        output = []

        def resolve_shift():
            mean = (min(output[idx - 1][0] + output[idx - 1][1] - output[shift_idx][0],
                        now - output[shift_idx][0] - 1) + output[idx - 1][0] - output[shift_idx][0] + 1) // 2
            dev = mean - (output[idx - 1][0] - output[shift_idx][0] + 1)
            output[shift_idx] = [output[shift_idx][0], self.get_random_number(mean, dev), vk_dict["shift"]]

        def resolve_ctrl_alt():
            mean1 = (min(output[idx - 1][0] + output[idx - 1][1] - output[ctrl_alt_idx][0], now -
                         output[ctrl_alt_idx][0] - 1) + output[idx - 1][0] - output[ctrl_alt_idx][0] + 1) // 2
            mean2 = (min(output[idx - 1][0] + output[idx - 1][1] - output[ctrl_alt_idx + 1][0], now -
                         output[ctrl_alt_idx + 1][0] - 1) + output[idx - 1][0] - output[ctrl_alt_idx + 1][0] + 1) // 2
            dev1 = mean1 - (output[idx - 1][0] - output[ctrl_alt_idx][0] + 1)
            dev2 = mean2 - (output[idx - 1][0] - output[ctrl_alt_idx + 1][0] + 1)
            ctrl_alt = [vk_dict["ctrl"], vk_dict["alt"]]
            self.rng.shuffle(ctrl_alt)
            output[ctrl_alt_idx] = [output[ctrl_alt_idx][0], self.get_random_number(mean1, dev1), ctrl_alt[0]]
            output[ctrl_alt_idx + 1] = [output[ctrl_alt_idx + 1][0], self.get_random_number(mean2, dev2), ctrl_alt[1]]

        for c in string:
            hold_latency = self.get_random_number(self.base_hold_latency, self.base_hold_deviation)
            press_latency = self.get_random_number(self.base_press_latency, self.base_press_deviation)
            if c.isupper() or c in shift_chars:
                if shift_idx == -1:
                    shift_idx = idx
                    output.append([now])
                    now += press_latency
                    idx += 1
                    press_latency = self.get_random_number(self.base_press_latency, self.base_press_deviation)

            elif c in ctrl_alt_chars:
                if ctrl_alt_idx == -1:
                    ctrl_alt_idx = idx
                    output.append([now])
                    now += press_latency // 4
                    output.append([now])
                    now += self.get_random_number(self.base_press_latency, self.base_press_deviation)
                    idx += 2
                    press_latency = self.get_random_number(self.base_press_latency, self.base_press_deviation)
            else:
                if shift_idx != -1:
                    resolve_shift()
                    shift_idx = -1
                elif ctrl_alt_idx != -1:
                    resolve_ctrl_alt()
                    ctrl_alt_idx = -1

            if (idx > 1 and len(output[idx - 1]) > 1 and vk_dict[c] == output[idx - 1][2]
                    and now <= output[idx - 1][0] + output[idx - 1][1]):
                now = output[idx - 1][0] + output[idx - 1][1] + self.get_random_number(self.base_press_deviation,
                                                                                       self.base_press_deviation)
            output.append([now, hold_latency, vk_dict[c]])
            now += press_latency
            idx += 1

        if shift_idx != -1:
            resolve_shift()
        elif ctrl_alt_idx != -1:
            resolve_ctrl_alt()
        return Tensor(output).to(dtype=torch.int)

    def get_random_number(self, mean: int, deviation: int) -> int:
        match self.distribution:
            case "normal":
                number = int(self.rng.gauss(mean, deviation / 4))
                while number not in range(mean - deviation, mean + deviation):
                    number = int(self.rng.gauss(mean, deviation / 4))
                return number
            case "uniform":
                return mean + self.rng.randint(-deviation, deviation)


if __name__ == '__main__':
    kg = KeystrokeGenerator()
    print(kg.generate_keystroke("HEllo @world!"))
