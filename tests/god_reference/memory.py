"""Store deterministic 32-bit game structures for upstream reader comparisons."""
import struct

from soku_rl.env.observation.memory_schema import FIGHTER_FIELDS, OBJECT_FIELDS, SPECIAL_FIELDS


class Memory:
    def __init__(self):
        self.regions = {0x800000: bytearray(0xA1000), 0x100000: bytearray(0x40000)}

    def region(self, address, size):
        for start, values in self.regions.items():
            if start <= address and address + size <= start + len(values):
                return values, address - start
        raise ValueError(f"unmapped reference address {address:#x}, length {size}")

    def read(self, address, size):
        values, index = self.region(address, size)
        return bytes(values[index:index + size])

    def begin_frame(self):
        pass

    def write(self, address, kind, *values):
        data = struct.pack("<" + kind, *values)
        region, index = self.region(address, len(data))
        region[index:index + len(data)] = data


def game_memory(character):
    memory = Memory()
    memory.write(0x8985E4, "I", 0x100000)
    memory.write(0x8985DC, "I", 0x100100)
    memory.write(0x100100 + 0x40, "2I", 0x100200, 0x100208)
    for seat in (0, 1):
        base = 0x101000 + seat * 0x10000
        memory.write(0x100000 + 12 + seat * 4, "I", base)
        memory.write(0x100200 + seat * 4, "I", base)
        memory.write(0x899D10 + seat * 0x20, "I", (character + seat) % 20)
        for name, (offset, kind) in (*OBJECT_FIELDS.items(), *FIGHTER_FIELDS.items()):
            value = 1.25 if kind == "f" else 1
            if name == "dir": value = 1 if seat == 0 else -1
            memory.write(base + offset, kind, value)
        for _, offset, kind in SPECIAL_FIELDS.values():
            memory.write(base + offset, kind, 2)
        memory.write(base + 0xEC, "2f", 400.25 + seat * 100, 20.5)
        memory.write(base + 0x150, "I", base + 0x1000)
        memory.write(base + 0x100A, "I", 32)
        memory.write(base + 0x104C, "2I", 0x80012345, 0xFFFFFFFF)
        memory.write(base + 0x1CB, "2B", 2, 1)
        memory.write(base + 0x1D0, "4i", 100, -20, 200, 30)
        memory.write(base + 0x220, "4i", 120, -40, 300, 40)
        memory.write(base + 0x320, "2I", 0, base + 0x1100)
        memory.write(base + 0x334, "I", 1)
        memory.write(base + 0x1100, "4i", -40, -60, 80, 70)
        memory.write(base + 0x5E4, "HB", 123, 2)
        memory.write(base + 0x5EC, "4I", base + 0x1200, 5, 4, 2)
        for index in range(5):
            memory.write(base + 0x1200 + index * 4, "I", base + 0x1300 + index * 4)
            memory.write(base + 0x1300 + index * 4, "2h", 100 + index, index + 1)
        memory.write(base + 0x6C4, "16b", *(list(range(4)) + [-1] * 12))
        memory.write(base + 0x750, "I", base + 0x1400)
        memory.write(base + 0x1400, "I", base + 0x1500)
        memory.write(base + 0x1538, "8i", -3, 4, 1, 2, 3, 4, 5, 6)
        memory.write(base + 0x6F8, "I", base + 0x1600)
        memory.write(base + 0x1658, "3I", 0, base + 0x1700, 1)
        memory.write(base + 0x1700, "3I", base + 0x1710, base + 0x1710, 0)
        memory.write(base + 0x1710, "3I", base + 0x1700, base + 0x1700, base + 0x2000)
        for offset, kind in OBJECT_FIELDS.values():
            memory.write(base + 0x2000 + offset, kind, 1)
        memory.write(base + 0x2150, "I", base + 0x1000)
        memory.write(0x899D18 + seat * 0x20, "5I", 0, base + 0x1800, 4, 3, 20)
        for chunk in range(4):
            memory.write(base + 0x1800 + chunk * 4, "I", base + 0x1900 + chunk * 16)
            memory.write(base + 0x1900 + chunk * 16, "8H", *range(chunk * 8, chunk * 8 + 8))
    return memory
