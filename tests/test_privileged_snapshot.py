"""Compare the native bounded capture with the existing complete-state reader."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from god_reference.memory import game_memory
from game_runtime.privileged import PrivilegedReader
from game_runtime.snapshot_memory import SnapshotMemory


class Region(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("address", "size", "offset")]


class Snapshot(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("regions", "used", "error")] + [
        ("index", Region * 40000), ("data", C.c_ubyte * (4 * 1024 * 1024))]


READ = C.CFUNCTYPE(C.c_bool, C.c_uint32, C.c_void_p, C.c_uint32)


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler required for native snapshot verification")
    native = Path(__file__).parents[1] / "native/SokuRLBridge"
    directory = tmp_path_factory.mktemp("privileged-snapshot")
    path = directory / ("capture.dll" if sys.platform == "win32" else "capture.so")
    flags = ["-static"] if sys.platform == "win32" else ["-fPIC"]
    subprocess.run([compiler, "-std=c++17", "-shared", "-O2", *flags,
        str(native / "PrivilegedSnapshot.cpp"), str(native / "tests/PrivilegedSnapshotReference.cpp"),
        "-o", str(path)], check=True, capture_output=True)
    library = C.CDLL(str(path))
    library.captureSnapshot.argtypes = [C.POINTER(Snapshot), READ]
    library.captureSnapshot.restype = None
    return library.captureSnapshot


def snapshot(capture, memory):
    errors = []

    @READ
    def read(address, destination, size):
        try:
            C.memmove(destination, memory.read(address, size), size)
            return True
        except Exception as error:
            errors.append(error)
            return False

    output = Snapshot()
    capture(C.byref(output), read)
    assert not errors, errors
    return output


@pytest.mark.parametrize("character", range(20))
@pytest.mark.parametrize("weather", [0, 2, 11])
def test_every_character_and_card_weather_matches_live_memory(capture, character, weather):
    memory = game_memory(character)
    memory.write(0x8971C0, "i", weather)
    direct = PrivilegedReader(memory)
    copied = PrivilegedReader(memory)
    for frame in range(3):
        output = snapshot(capture, memory)
        assert output.error == 0
        regions = [(r.address, r.size, r.offset) for r in output.index[:output.regions]]
        copied.memory = SnapshotMemory(regions, bytes(output.data[:output.used]))
        raw = SimpleNamespace(frameId=frame, segmentId=7)
        client = SimpleNamespace(snapshot=lambda: SimpleNamespace(
            run_state_name="PAUSED", game_frame=frame, latest=raw))
        assert copied.observe(raw, client) == direct.observe(raw, client)
        before = copied.memory.read(0x101000 + 0xEC, 4)
        memory.write(0x101000 + 0xEC, "f", 500.5 + frame)
        assert copied.memory.read(0x101000 + 0xEC, 4) == before


def test_invalid_object_count_stops_capture(capture):
    memory = game_memory(0)
    memory.write(0x101000 + 0x1600 + 0x60, "I", 1025)
    assert snapshot(capture, memory).error == 3


def test_invalid_pointer_returns_read_error_without_throwing(capture):
    output = Snapshot()
    capture(C.byref(output), READ(lambda address, destination, size: False))
    assert output.error == 1 and output.regions == 0 and output.used == 0


def test_snapshot_rejects_uncaptured_or_inconsistent_memory():
    memory = SnapshotMemory([(10, 4, 0), (11, 1, 1)], b"abcd")
    assert memory.read(10, 4) == b"abcd"
    assert memory.read(11, 3) == b"bcd"
    with pytest.raises(ValueError, match="uncaptured"):
        memory.read(9, 1)
    with pytest.raises(ValueError, match="inconsistent"):
        SnapshotMemory([(10, 1, 0), (10, 1, 1)], b"ab")
