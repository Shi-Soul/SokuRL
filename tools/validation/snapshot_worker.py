"""Compare native captures with direct memory on the same paused game frames."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from game_runtime import batch
from game_runtime.observation import ObservationReader
from game_runtime.offline_snapshot import offline_snapshot_requested
from game_runtime.privileged import PrivilegedReader, ProcessMemory
from soku_rl.env.observation.privileged import encode_privileged
import rollout_worker


class VerifiedObservationReader(ObservationReader):
    def __init__(self, pid, mode, visibility):
        if not offline_snapshot_requested() or mode != 'privileged_state':
            raise ValueError('snapshot verification requires the native privileged transport')
        super().__init__(pid, mode, visibility)
        self.reference = PrivilegedReader(ProcessMemory(pid))
        self.resources.append(self.reference)
        self.digest = hashlib.sha256()
        self.frames = 0
        self.initial_frames = 0
        self.characters = set()
        self.weathers = set()
        self.max_objects = 0
        self.report = Path(os.environ['SOKURL_SNAPSHOT_AUDIT']) / f'pid-{pid}.json'
        self.report.parent.mkdir(parents=True, exist_ok=True)

    def read(self, raw, bridge):
        actual = super().read(raw, bridge)
        reference = self.reference.observe(raw, bridge)
        if actual.observations != reference:
            raise AssertionError(f'complete privileged state differs at frame {raw.frameId}')
        for left, right in zip(actual.observations, reference, strict=True):
            encoded, expected = encode_privileged(left), encode_privileged(right)
            if not np.array_equal(encoded.view(np.uint32), expected.view(np.uint32)):
                raise AssertionError(f'encoded observation bits differ at frame {raw.frameId}')
            self.digest.update(encoded.tobytes())
        self.frames += 1
        self.initial_frames += int(raw.frameId == 0)
        self.characters.add(tuple(p['char'] for p in reference[0].players))
        self.weathers.add(reference[0].world['weather'])
        self.max_objects = max(self.max_objects, *(p['obj_n'] for p in reference[0].players))
        return actual

    def close(self):
        super().close()
        self.report.write_text(json.dumps({'pid': self.pid, 'verified_frames': self.frames,
            'initial_frames': self.initial_frames, 'characters': sorted(self.characters),
            'weathers': sorted(self.weathers), 'maximum_objects': self.max_objects,
            'observation_bytes_sha256': self.digest.hexdigest()}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    batch.ObservationReader = VerifiedObservationReader
    rollout_worker.main()
