"""Use the same paused-frame observations for live environments and replays."""
from soku_rl.env.observation.diagnostic import observe
from soku_rl.env.observation.visible_state import observe_visible_states
from soku_rl.pomg import Outcome, TimeStep


def time_step(raw, dropped_frames, observations, pid):
    if raw.sceneId != 5 or raw.battleMode != 3:
        raise RuntimeError("game left VS battle")
    if dropped_frames:
        raise RuntimeError("frame recording was incomplete")
    hp = raw.p1.hp, raw.p2.hp
    if hp[0] <= 0 and hp[1] <= 0:
        outcome, rewards = Outcome.DRAW, (0.0, 0.0)
    elif hp[0] <= 0:
        outcome, rewards = Outcome.P2_WIN, (-1.0, 1.0)
    elif hp[1] <= 0:
        outcome, rewards = Outcome.P1_WIN, (1.0, -1.0)
    else:
        outcome, rewards = Outcome.ONGOING, (0.0, 0.0)
    return TimeStep(raw.frameId, observations, rewards, outcome, {
        "pid": pid, "segment": raw.segmentId,
        "hp": hp, "characters": (raw.p1.characterId, raw.p2.characterId),
        "stage": raw.stageId, "weather": raw.activeWeather,
        "hash": f"{raw.stateHash:016X}", "dropped_frames": dropped_frames,
        "objects": (raw.p1ObjectCount, raw.p2ObjectCount),
    })


class ObservationReader:
    def __init__(self, pid, mode, visibility):
        if mode not in {"privileged_state", "image", "state", "diagnostic_state"}:
            raise ValueError("unsupported observation mode")
        self.pid, self.mode, self.visibility = pid, mode, visibility
        self.resources = []
        if mode == "privileged_state":
            from privileged_reader import PrivilegedReader, ProcessMemory
            self.privileged = PrivilegedReader(ProcessMemory(pid))
            self.resources.append(self.privileged)
        if mode in {"image", "state"}:
            from image_shared import ImageClient
            self.image = ImageClient(pid)
            self.resources.append(self.image)

    def read(self, raw, bridge):
        if self.mode == "privileged_state":
            observations = self.privileged.observe(raw, bridge)
        elif self.mode == "image":
            scene = self.image.read(int(raw.frameId), 10.0)
            observations = (scene.image, scene.image)
        elif self.mode == "state":
            render = self.image.read_state(int(raw.frameId), 10.0)
            observations = observe_visible_states(raw, render, self.visibility)
        else:
            observations = tuple(observe(raw, seat) for seat in (0, 1))
        return time_step(raw, bridge.snapshot().dropped_frames, observations, self.pid)

    def close(self):
        errors = []
        while self.resources:
            try:
                self.resources.pop().close()
            except Exception as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError(f"cannot close observation readers: {errors}")
