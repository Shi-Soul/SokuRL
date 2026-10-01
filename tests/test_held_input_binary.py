"""Execute the built adapter and original input decoder in an isolated x86 emulator."""
import hashlib
from pathlib import Path
import struct

import pytest

pytest.importorskip("unicorn")
from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_EIP, UC_X86_REG_ESP

ROOT = Path(__file__).parents[1]


def map_image(cpu, path):
    data = path.read_bytes()
    pe, = struct.unpack_from("<I", data, 0x3C)
    count, = struct.unpack_from("<H", data, pe + 6)
    optional_size, = struct.unpack_from("<H", data, pe + 20)
    optional = pe + 24
    assert struct.unpack_from("<H", data, optional)[0] == 0x10B
    base, = struct.unpack_from("<I", data, optional + 28)
    size, = struct.unpack_from("<I", data, optional + 56)
    cpu.mem_map(base, (size + 4095) & ~4095)
    for index in range(count):
        _, _, rva, raw_size, offset = struct.unpack_from("<8sIIII", data, optional + optional_size + index * 40)
        if raw_size:
            cpu.mem_write(base + rva, data[offset:offset + raw_size])
    export, = struct.unpack_from("<I", data, optional + 96)
    imports, = struct.unpack_from("<I", data, optional + 104)
    memset_slots = []
    if imports:
        descriptor = base + imports
        while True:
            original, _, _, name, first = struct.unpack("<5I", cpu.mem_read(descriptor, 20))
            if not name:
                break
            index = 0
            while True:
                entry, = struct.unpack("<I", cpu.mem_read(base + (original or first) + index * 4, 4))
                if not entry:
                    break
                if not entry & 0x80000000:
                    name = bytes(cpu.mem_read(base + entry + 2, 64)).split(b"\0", 1)[0]
                    if name == b"memset":
                        memset_slots.append(base + first + index * 4)
                index += 1
            descriptor += 20
    return base, base + export, memset_slots


class Decoder:
    def __init__(self, cpu, function, memset_slots):
        self.cpu, self.function = cpu, function
        cpu.mem_map(0x20000000, 0x30000)
        # Resolve only the adapter's C-runtime zero-fill. All input handling
        # still executes the compiled adapter and unchanged original game code.
        for slot in memset_slots:
            cpu.mem_write(slot, struct.pack("<I", 0x2002E000))
        cpu.hook_add(UC_HOOK_CODE, self.memset, begin=0x2002E000, end=0x2002E000)

    def memset(self, cpu, address, size, context):
        stack = cpu.reg_read(UC_X86_REG_ESP)
        returned, target, value, count = struct.unpack("<4I", cpu.mem_read(stack, 16))
        cpu.mem_write(target, bytes([value & 255]) * count)
        cpu.reg_write(UC_X86_REG_EAX, target)
        cpu.reg_write(UC_X86_REG_ESP, stack + 4)
        cpu.reg_write(UC_X86_REG_EIP, returned)

    def __call__(self, previous, intent):
        cpu = self.cpu
        cpu.mem_write(0x20010000, struct.pack("<8i", *previous))
        cpu.mem_write(0x20010100, struct.pack("<8i", *intent))
        cpu.mem_write(0x20010200, bytes(32))
        cpu.mem_write(0x20008000, struct.pack("<4I", 0x2002F000, 0x20010000, 0x20010100, 0x20010200))
        cpu.reg_write(UC_X86_REG_ESP, 0x20008000)
        cpu.emu_start(self.function, 0x2002F000, count=20000)
        assert cpu.reg_read(UC_X86_REG_EIP) == 0x2002F000
        return struct.unpack("<8i", cpu.mem_read(0x20010200, 32))


@pytest.fixture(scope="module")
def decoder():
    game = ROOT / "th123_jp/th123.exe"
    adapter = ROOT / "native/SokuRLBridge/build/HeldInputReference.dll"
    if not game.is_file() or not adapter.is_file():
        pytest.skip("original th123 1.10a and the MSVC x86 HeldInputReference build are required")
    assert hashlib.md5(game.read_bytes()).hexdigest() == "df35d1fbc7b583317adabe8cd9f53b2e"
    cpu = Uc(UC_ARCH_X86, UC_MODE_32)
    assert map_image(cpu, game)[0] == 0x400000
    base, export, memset_slots = map_image(cpu, adapter)
    count, functions, names, ordinals = struct.unpack("<4I", cpu.mem_read(export + 24, 16))
    for index in range(count):
        name_rva, = struct.unpack("<I", cpu.mem_read(base + names + index * 4, 4))
        name = bytes(cpu.mem_read(base + name_rva, 64)).split(b"\0", 1)[0]
        if name.lstrip(b"_") == b"decodeInput":
            ordinal, = struct.unpack("<H", cpu.mem_read(base + ordinals + index * 2, 2))
            address, = struct.unpack("<I", cpu.mem_read(base + functions + ordinal * 4, 4))
            return Decoder(cpu, base + address, memset_slots)
    raise AssertionError("compiled adapter export is missing")


@pytest.mark.parametrize("previous", [(0,) * 8, (7,) * 8, (-7, -7, 7, 0, 3, 0, 9, 0)])
def test_all_logical_combinations_use_original_held_counts(decoder, previous):
    for horizontal in (-1, 0, 1):
        for vertical in (-1, 0, 1):
            for mask in range(64):
                intent = (horizontal, vertical, *(int(bool(mask & (1 << bit))) for bit in range(6)))
                expected = tuple(0 if direction == 0 else
                    old + direction if old * direction > 0 else direction
                    for old, direction in zip(previous[:2], intent[:2], strict=True))
                expected += tuple(old + 1 if pressed else 0
                    for old, pressed in zip(previous[2:], intent[2:], strict=True))
                assert decoder(previous, intent) == expected, (previous, intent)
