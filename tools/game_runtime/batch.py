"""Adapt owned th123 processes to the simultaneous two-player game contract."""
import ctypes

from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.match import MatchConfig
from bridge_shared import BridgeClient, FRAME_RING_CAPACITY, wait_for_steps
from game_runtime.frames import FRAME_SIZE, drain_frames_into, wait_for_frame_zero
import sokurl
from game_runtime.observation import ObservationReader


RESET_METHODS = {"image": "process_restart", "state": "native_scene_reload",
                 "diagnostic_state": "native_scene_reload", "privileged_state": "native_scene_reload"}


class SokuGameBatch:
    def __init__(self, launch_timeout):
        if launch_timeout <= 0:
            raise ValueError("launch timeout must be positive")
        self.launch_timeout = launch_timeout
        self.processes = {}
        self.clients = {}
        self.buffers = {}
        self.frames = {}
        self.active = set()
        self.readers = {}
        self.observation_mode = "diagnostic_state"
        self.recording_enabled = False
        self.recordings = {}
        self.completed_replays = []

    def enable_recording(self):
        if self.processes:
            raise RuntimeError("enable replay recording before launching a game")
        self.recording_enabled = True

    def _finish_recording(self, slot, reason):
        if slot in self.recordings:
            recording = self.recordings.pop(slot)
            self.completed_replays.append(recording.finish(self.clients[slot], reason))

    def take_replays(self):
        completed, self.completed_replays = self.completed_replays, []
        return completed

    def configure_observation(self, configuration):
        mode = configuration["mode"]
        if self.processes or mode not in RESET_METHODS:
            raise ValueError("set a supported observation mode before launching games")
        self.observation_mode = mode
        self.visibility = VisibilityConfig(**configuration["visibility"])
        self.match = MatchConfig(**configuration["match"])
        self.max_frames = configuration["max_frames"]
        if type(self.max_frames) is not int or self.max_frames < 1:
            raise ValueError("a positive episode frame limit is required")

    def _observe(self, slot, raw, dropped):
        if dropped:
            raise RuntimeError("frame recording was incomplete")
        return self.readers[slot].read(raw, self.clients[slot])

    def reset(self, seeds):
        if self.processes or not seeds:
            raise ValueError("legacy batch reset requires a fresh batch")
        return self.reset_slots(dict(enumerate(seeds)))

    def reset_slots(self, seeds):
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("nonempty nonnegative slot IDs are required")
        if any(type(s) is not int or not 0 <= s < 0xFFFFFFFF for s in seeds.values()):
            raise ValueError("native seed 0xFFFFFFFF is reserved; use a smaller uint32")
        for slot in seeds:
            self._finish_recording(slot, "reset")
        if RESET_METHODS[self.observation_mode] == "process_restart":
            # Scene reload preserves renderer state that changes pixels across
            # episodes. Recreate only the selected image slots; state-only
            # observations have passed native scene-reload determinism checks.
            self._close_slots(set(seeds) & set(self.processes))
        existing = set(seeds) & set(self.processes)
        fresh = {slot: seed for slot, seed in seeds.items() if slot not in existing}
        self.active.difference_update(seeds)
        pending = {}
        for slot in sorted(existing):
            client = self.clients[slot]
            segment = (client.snapshot().latest.segmentId + 1) & 0xFFFFFFFF
            pending[slot] = (client.reset_episode(seeds[slot]), segment)
        states = self._launch_slots(fresh) if fresh else {}
        for slot, (sequence, segment) in pending.items():
            try:
                raw = self.clients[slot].wait_for_reset(sequence, segment, seeds[slot], self.launch_timeout)
            except TimeoutError as error:
                pid = self.processes[slot].pid
                try:
                    live = sokurl._read_process_values(pid)
                    detail = f"live_scene={live[0]} live_mode={live[1]} characters={live[2:4]}"
                except OSError as read_error:
                    detail = f"live process read failed: {read_error!r}"
                raise TimeoutError(f"{error}; {detail}") from error
            self.frames[slot] = 0
            states[slot] = self._observe(slot, raw, 0)
        self.active.update(seeds)
        if self.recording_enabled:
            from game_runtime.recording import EpisodeRecording
            for slot, seed in seeds.items():
                self.recordings[slot] = EpisodeRecording(self.processes[slot].pid, slot, seed)
        return states

    def _launch_slots(self, seeds):
        processes = sokurl._launch_vs_group_from_title(
            len(seeds), self.launch_timeout, headless=True, unlimited=True,
            seeds=tuple(seeds.values()), pause_at_start=True,
            capture_images=self.observation_mode == "image",
            capture_state=self.observation_mode == "state",
            match=self.match,
        )
        self.processes.update(zip(seeds, processes, strict=True))
        states = {}
        try:
            for slot, process in zip(seeds, processes, strict=True):
                client = BridgeClient(process.pid)
                self.clients[slot] = client
                raw = wait_for_frame_zero(client, process.pid, self.launch_timeout)
                self.buffers[slot] = (ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))()
                self.frames[slot] = 0
                self.readers[slot] = ObservationReader(process.pid, self.observation_mode, self.visibility)
                states[slot] = self._observe(slot, raw, 0)
                self.active.add(slot)
        except BaseException:
            self._close_slots(set(seeds))
            raise
        return states

    def step(self, actions):
        if not actions or not set(actions) <= self.active:
            raise ValueError("joint actions must refer to active games")
        slots = list(actions)
        for joint in actions.values():
            if len(joint) != 2:
                raise ValueError("two actions are required")
            for action in joint:
                values = action.inputs
                if (len(values) != 8 or any(type(v) is not int for v in values)
                        or any(v not in (-1, 0, 1) for v in values[:2])
                        or any(v not in (0, 1) for v in values[2:])):
                    raise ValueError("invalid logical input")
        sequences = [self.clients[s].step_with_inputs(actions[s][0].inputs, actions[s][1].inputs)
                     for s in slots]
        for slot in slots:
            self.frames[slot] += 1
        snapshots = wait_for_steps([self.clients[s] for s in slots], sequences,
                                   [self.frames[s] for s in slots], 10.0)
        states = {}
        for slot, snapshot in zip(slots, snapshots, strict=True):
            states[slot] = self._observe(slot, snapshot.latest, snapshot.dropped_frames)
            drain_frames_into(self.clients[slot], self.buffers[slot])
            if states[slot].ended:
                self._finish_recording(slot, states[slot].outcome.value)
            elif self.recording_enabled and states[slot].frame >= self.max_frames:
                self._finish_recording(slot, "time_limit")
        self.active.difference_update(s for s, state in states.items() if state.ended)
        return states

    def close(self):
        self._close_slots(set(self.processes))

    def _close_slots(self, slots):
        errors = []
        for slot in slots:
            try:
                self._finish_recording(slot, "close")
            except Exception as error:
                errors.append(repr(error))
            for resources in (self.readers, self.clients):
                try:
                    if slot in resources:
                        resources.pop(slot).close()
                except Exception as error:
                    errors.append(repr(error))
            try:
                process = self.processes[slot]
                if process.is_running():
                    sokurl.shutdown(5.0, process.pid)
                del self.processes[slot]
            except Exception as error:
                errors.append(repr(error))
            self.buffers.pop(slot, 0)
            self.frames.pop(slot, 0)
            self.active.discard(slot)
        if errors:
            raise RuntimeError(f"failed to close owned game resources: {errors}")
