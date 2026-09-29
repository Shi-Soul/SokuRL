"""Store padded numeric observations without losing any array bits."""
from dataclasses import dataclass
import zlib

import numpy as np


@dataclass(frozen=True)
class PackedObservation:
    shape: tuple
    dtype: str
    content: bytes

    @classmethod
    def pack(cls, observation):
        values = np.asarray(observation)
        if values.dtype.hasobject:
            raise TypeError("observation storage requires a numeric array")
        return cls(values.shape, values.dtype.str, zlib.compress(values.tobytes(), 1))

    def unpack(self):
        return np.frombuffer(zlib.decompress(self.content), dtype=self.dtype).reshape(self.shape).copy()


class PackedArray:
    """Materialize only the sampled observations; retain upstream buffer indexing."""
    def __init__(self, samples, observation_shape, dtype):
        self.samples, self.observation_shape, self.dtype = samples, tuple(observation_shape), np.dtype(dtype)

    @property
    def shape(self):
        return (*self.samples.shape, *self.observation_shape)

    def __setitem__(self, key, values):
        shape = np.shape(self.samples[key])
        if values.shape != (*shape, *self.observation_shape) or values.dtype != self.dtype:
            raise ValueError("rollout observation shape or dtype changed")
        packed = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            packed[index] = PackedObservation.pack(values[index])
        self.samples[key] = packed

    def __getitem__(self, key):
        selected = np.asarray(self.samples[key], dtype=object)
        result = np.empty((*selected.shape, *self.observation_shape), dtype=self.dtype)
        for index in np.ndindex(selected.shape):
            result[index] = selected[index].unpack()
        return result

    def swapaxes(self, first, second):
        return PackedArray(self.samples.swapaxes(first, second), self.observation_shape, self.dtype)

    def reshape(self, *shape):
        if tuple(shape[-len(self.observation_shape):]) != self.observation_shape:
            raise ValueError("buffer flattening must preserve each observation")
        return PackedArray(self.samples.reshape(shape[:-len(self.observation_shape)]),
                           self.observation_shape, self.dtype)
