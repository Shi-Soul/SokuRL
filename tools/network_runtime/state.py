"""Read private, latest-only network telemetry without pausing the game."""
import ctypes
from dataclasses import dataclass
import struct
import time

import bridge_shared
from soku_rl.env.observation.render_state import RENDER_STATE_SIZE, RenderSnapshot
from soku_rl.play.match import MatchState


HEADER = struct.Struct("<8IQ2I")
RAW_SIZE = ctypes.sizeof(bridge_shared.RawFrameState)
MAPPING_SIZE = HEADER.size + RAW_SIZE + RENDER_STATE_SIZE
MAGIC = 0x54454E53


@dataclass(frozen=True)
class NetworkSnapshot:
    connected: bool
    scene: int
    match: int
    local_seat: int
    updates: int
    scores: tuple[int, int]
    raw: bridge_shared.RawFrameState
    render: RenderSnapshot

    @property
    def in_battle(self):
        return self.connected and self.scene in (13, 14) and self.updates > 0

    @property
    def match_state(self):
        if not self.connected or self.scene not in (8, 9, 10, 11, 13, 14):
            phase = "disconnected"
        elif self.in_battle:
            phase = "battle"
        else:
            phase = "menu" if self.scene in (8, 9) else "loading"
        return MatchState(self.match, self.raw.roundId, self.updates, self.scores,
                          (self.raw.p1.hp, self.raw.p2.hp), phase)


def decode_network_state(data):
    if len(data) != MAPPING_SIZE:
        raise ValueError("invalid network mapping length")
    magic, version, size, sequence, connected, scene, match, seat, updates, left, right = HEADER.unpack_from(data)
    if (magic, version, size) != (MAGIC, 2, MAPPING_SIZE):
        raise ValueError("unsupported network mapping ABI")
    if sequence & 1 or connected not in (0, 1) or seat not in (0, 1, 0xFFFFFFFF):
        raise ValueError("invalid network mapping header")
    raw = bridge_shared.RawFrameState.from_buffer_copy(data, HEADER.size)
    render = RenderSnapshot.decode(data[HEADER.size + RAW_SIZE:])
    if scene in (13, 14) and updates:
        if (raw.frameId, raw.segmentId, raw.sceneId) != (updates, match, scene):
            raise ValueError("network frame identity differs from its header")
        if raw.stateHash != bridge_shared.calculate_state_hash(raw):
            raise ValueError("network state checksum differs")
    return NetworkSnapshot(bool(connected), scene, match, seat, updates, (left, right), raw, render)


class NetworkStateClient:
    def __init__(self, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("a positive game PID is required")
        kernel = bridge_shared._kernel32
        self.handle = kernel.OpenFileMappingW(4, False, rf"Local\SokuRLNetwork_{pid}")
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "network mapping is unavailable")
        self.view = kernel.MapViewOfFile(self.handle, 4, 0, 0, MAPPING_SIZE)
        if not self.view:
            error = ctypes.get_last_error()
            kernel.CloseHandle(self.handle)
            self.handle = None
            raise OSError(error, "cannot map network state")

    def read(self, timeout):
        if not self.view or timeout <= 0:
            raise ValueError("read requires an open mapping and positive timeout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            first = ctypes.c_uint32.from_address(self.view + 12).value
            if not first & 1:
                data = ctypes.string_at(self.view, MAPPING_SIZE)
                last = ctypes.c_uint32.from_address(self.view + 12).value
                if first == last == struct.unpack_from("<I", data, 12)[0]:
                    return decode_network_state(data)
            time.sleep(.001)
        raise TimeoutError("could not copy a stable network state")

    def close(self):
        kernel = bridge_shared._kernel32
        if self.view:
            kernel.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            kernel.CloseHandle(self.handle)
            self.handle = None
