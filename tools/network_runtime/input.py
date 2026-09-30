"""Submit local netplay keys; injection time is distinct from engine execution."""
import ctypes
import struct
import time

import bridge_shared


SIZE = 104
MAGIC = 0x494E4B53
BODY = struct.Struct("<4I2Q8i")
RESULTS = {1: "accepted", 2: "released", 3: "invalid", 4: "wrong_round",
           5: "late", 6: "too_frequent", 7: "queue_full"}


def input_payload(snapshot, keys, duration):
    if (not snapshot.in_battle or max(snapshot.scores) >= 2 or
            min(snapshot.raw.p1.hp, snapshot.raw.p2.hp) <= 0):
        raise ValueError("network input requires an active battle snapshot")
    if type(duration) is not int or not 1 <= duration <= 120:
        raise ValueError("hold duration must be between 1 and 120 frames")
    if len(keys) != 8 or any(type(value) is not int for value in keys):
        raise ValueError("eight integer keys are required")
    if any(value not in (-1, 0, 1) for value in keys[:2]) or any(value not in (0, 1) for value in keys[2:]):
        raise ValueError("invalid axes or buttons")
    return BODY.pack(1, snapshot.match, snapshot.raw.roundId, duration,
                     snapshot.updates, snapshot.updates + 5, *keys)


def result_confirm_payload(snapshot):
    if not snapshot.in_battle or max(snapshot.scores) < 2:
        raise ValueError("result confirmation requires a completed match")
    return BODY.pack(3, snapshot.match, snapshot.raw.roundId, 1,
                     snapshot.updates, snapshot.updates + 5, 0, 0, 1, 0, 0, 0, 0, 0)


class NetworkInputClient:
    def __init__(self, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("a positive game PID is required")
        kernel = bridge_shared._kernel32
        self.handle = kernel.OpenFileMappingW(6, False, rf"Local\SokuRLNetworkInput_{pid}")
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "network input mapping is unavailable")
        self.view = kernel.MapViewOfFile(self.handle, 6, 0, 0, SIZE)
        if not self.view:
            error = ctypes.get_last_error()
            kernel.CloseHandle(self.handle)
            self.handle = None
            raise OSError(error, "cannot map network input")
        if struct.unpack("<3I", ctypes.string_at(self.view, 12)) != (MAGIC, 1, SIZE):
            self.close()
            raise ValueError("unsupported network input ABI")

    def _send(self, data):
        if not self.view:
            raise RuntimeError("network input mapping is closed")
        request, ack = struct.unpack("<2I", ctypes.string_at(self.view + 12, 8))
        if request != ack:
            raise RuntimeError("wait for the previous request acknowledgment")
        sequence = (request + 1) & 0xFFFFFFFF or 1
        ctypes.memmove(self.view + 24, data, BODY.size)
        ctypes.c_uint32.from_address(self.view + 12).value = sequence
        return sequence

    def submit(self, snapshot, keys, duration):
        return self._send(input_payload(snapshot, keys, duration))

    def release(self):
        return self._send(BODY.pack(2, 0, 0, 0, 0, 0, *([0] * 8)))

    def confirm_result(self, snapshot):
        return self._send(result_confirm_payload(snapshot))

    def wait_for_reply(self, sequence, timeout):
        if timeout <= 0:
            raise ValueError("reply requires a positive timeout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.read_reply(sequence)
            if result != "pending":
                return result
            time.sleep(.001)
        raise TimeoutError("network input acknowledgment timed out")

    def read_reply(self, sequence):
        if not self.view or type(sequence) is not int or sequence < 1:
            raise ValueError("reply requires an open mapping and a positive sequence")
        ack, result = struct.unpack("<2I", ctypes.string_at(self.view + 16, 8))
        if ack != sequence:
            return "pending"
        if result not in RESULTS:
            raise RuntimeError("unknown network input result")
        return RESULTS[result]

    def latest_injection(self, timeout):
        if not self.view or timeout <= 0:
            raise ValueError("injection read requires an open mapping and positive timeout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            before = ctypes.c_uint32.from_address(self.view + 88).value
            data = ctypes.string_at(self.view + 88, 16)
            after = ctypes.c_uint32.from_address(self.view + 88).value
            sequence, request, frame = struct.unpack("<IIQ", data)
            if before == after == sequence and not sequence & 1:
                return request, frame
            time.sleep(.001)
        raise TimeoutError("network input injection snapshot timed out")

    def close(self):
        # Release is an explicit acknowledged command, not an unverified close side effect.
        kernel = bridge_shared._kernel32
        if self.view:
            kernel.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            kernel.CloseHandle(self.handle)
            self.handle = None
