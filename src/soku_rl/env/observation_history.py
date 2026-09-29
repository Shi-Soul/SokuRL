"""Stack consecutive public observations for training and live inference."""
from collections import deque

import numpy as np

from soku_rl.pixels import RGBFrame
from soku_rl.visible_state import StateObservation
from .encoding import AGENTS


class ObservationHistory:
    def __init__(self, config):
        self.config = config
        self.history = [deque(maxlen=config.history_frames) for _ in AGENTS]
        self.frame = -1

    def _encode(self, frame, observations):
        if type(frame) is not int or frame < 0 or len(observations) != len(AGENTS):
            raise ValueError("a nonnegative frame and both public observations are required")
        for observation in observations:
            if isinstance(observation, (RGBFrame, StateObservation)) and observation.frame != frame:
                raise RuntimeError("image and simulation frame do not match")
        return tuple(self.config.encode(observation) for observation in observations)

    def reset(self, frame, observations):
        encoded = self._encode(frame, observations)
        for history, value in zip(self.history, encoded, strict=True):
            history.clear()
            history.extend(value.copy() for _ in range(self.config.history_frames))
        self.frame = frame

    def append(self, frame, observations):
        if self.frame < 0 or frame != self.frame + 1:
            raise RuntimeError("observation history requires consecutive frames after reset")
        encoded = self._encode(frame, observations)
        for history, value in zip(self.history, encoded, strict=True):
            history.append(value)
        self.frame = frame

    def clear(self):
        for history in self.history:
            history.clear()
        self.frame = -1

    def observations(self):
        if self.frame < 0:
            raise RuntimeError("reset observation history before reading it")
        result = {agent: np.concatenate(self.history[index]) for index, agent in enumerate(AGENTS)}
        if self.config.observation_mode == "image":
            for index, agent in enumerate(AGENTS):
                role = np.full((1, 240, 320), index * 255, dtype=np.uint8)
                result[agent] = np.concatenate((result[agent], role))
        return result

    def image(self, seat):
        if self.frame < 0 or self.config.observation_mode != "image" or type(seat) is not int or seat not in (0, 1):
            raise ValueError("image requires a reset image history and a valid seat")
        return self.history[seat][-1].transpose(1, 2, 0).copy()
