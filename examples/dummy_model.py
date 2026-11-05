"""A silent model for checking the interface without loading model weights."""

import numpy as np


class SilentModel:
    sr = 24000
    out_sr = 24000

    def feed(self, audio):
        return np.zeros(len(audio), dtype="float32")


def factory(config):
    # A real integration may load shared weights here. The returned callable
    # must create a fresh conversation state each time the evaluator calls it.
    return SilentModel
