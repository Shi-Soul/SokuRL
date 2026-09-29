"""Execute pinned package instructions in an isolated x86 emulator for auditing."""
import hashlib
from pathlib import Path
import re
import struct

from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP


PACKAGE_SHA256 = "ff8fb443c227c0eeb0e1ff6f92ec0afed965d4f95e07f35247c3fb8579ab862f"


class BinaryReference:
    """Supply read-only game memory and address settings to original instructions."""
    def __init__(self, executable, address_header, ini, memory):
        data = Path(executable).read_bytes()
        if hashlib.sha256(data).hexdigest() != PACKAGE_SHA256:
            raise ValueError("binary reference requires the exact audited executable")
        self.cpu = Uc(UC_ARCH_X86, UC_MODE_32)
        self.memory = memory
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        sections = struct.unpack_from("<H", data, pe + 6)[0]
        optional = pe + 24
        optional_size = struct.unpack_from("<H", data, pe + 20)[0]
        base = struct.unpack_from("<I", data, optional + 28)[0]
        size = struct.unpack_from("<I", data, optional + 56)[0]
        if base != 0x400000:
            raise ValueError("unexpected package image base")
        self.cpu.mem_map(base, (size + 4095) & ~4095)
        for i in range(sections):
            _, _, rva, raw_size, offset = struct.unpack_from("<8sIIII", data, optional + optional_size + i * 40)
            if raw_size:
                self.cpu.mem_write(base + rva, data[offset:offset + raw_size])
        self.cpu.mem_map(0x600000, 0x20000)
        self.cpu.mem_map(0x700000, 0x10000)
        # Object virtual calls have no game-state behavior for an Obj instance.
        self.cpu.mem_write(0x610000, b"\xc2\x04\x00")
        self.write(0x600000, "2I", 0x610000, 0x610000)
        self.write(0x44E084, "I", 0x610100)  # ReadProcessMemory import slot.
        header = Path(address_header).read_bytes()
        names = re.search(rb"enum\s*\{(.*?)ADDR_MAX", header, re.S).group(1)
        names = re.findall(rb"\b[A-Z][A-Z0-9_]+\b", names)
        values = dict(re.findall(rb"SWRS_ADDR_(\w+)\s*=\s*(0x[0-9A-Fa-f]+)", Path(ini).read_bytes()))
        self.addresses = {i: int(values[name], 16) for i, name in enumerate(names) if name in values}
        self.callbacks = {0x42A790: self.address, 0x610100: self.read_memory,
                          0x401240: self.lua_count, 0x401770: self.lua_is_number,
                          0x401A20: self.lua_integer, 0x401DC0: self.lua_push,
                          0x4269C0: self.object_pointer}
        for address in self.callbacks:
            self.cpu.hook_add(UC_HOOK_CODE, self.dispatch, begin=address, end=address)

    def write(self, address, kind, *values):
        self.cpu.mem_write(address, struct.pack("<" + kind, *values))

    def read(self, address, kind):
        return struct.unpack("<" + kind, self.cpu.mem_read(address, struct.calcsize("<" + kind)))

    def arguments(self, count):
        return self.read(self.cpu.reg_read(UC_X86_REG_ESP) + 4, "I" * count)

    def return_value(self, value, pop_bytes):
        stack = self.cpu.reg_read(UC_X86_REG_ESP)
        self.cpu.reg_write(UC_X86_REG_EAX, value)
        self.cpu.reg_write(UC_X86_REG_EIP, self.read(stack, "I")[0])
        self.cpu.reg_write(UC_X86_REG_ESP, stack + 4 + pop_bytes)

    def dispatch(self, cpu, address, size, context):
        self.callbacks[address]()

    def address(self):
        index, = self.arguments(1)
        self.write(0x610200, "I", self.addresses[index])
        self.return_value(0x610200, 4)

    def read_memory(self):
        _, source, target, size, count = self.arguments(5)
        data = self.memory.read(source, size)
        self.cpu.mem_write(target, data)
        if count:
            self.write(count, "I", len(data))
        self.return_value(1, 20)

    def lua_count(self):
        self.return_value(len(self.lua_arguments), 0)

    def lua_is_number(self):
        self.return_value(1, 0)

    def lua_integer(self):
        _, index = self.arguments(2)
        self.return_value(self.lua_arguments[index - 1], 0)

    def lua_push(self):
        stack = self.cpu.reg_read(UC_X86_REG_ESP)
        self.lua_results.append(self.read(stack + 8, "d")[0])
        self.return_value(0, 0)

    def object_pointer(self):
        index, = self.arguments(1)
        self.return_value(0x602000 if index == 0 else 0, 4)

    def object_values(self, game_address):
        self.entity(game_address)
        self.lua_arguments, self.lua_results = (0, 0), []
        count = self.call(0x425590, 0, (0,))
        if count != len(self.lua_results):
            raise RuntimeError("packaged Lua return count differs from captured values")
        return tuple(self.lua_results)

    def call(self, function, this, arguments):
        self.write(0x708000, "I" * (len(arguments) + 1), 0x611000, *arguments)
        self.cpu.reg_write(UC_X86_REG_ESP, 0x708000)
        self.cpu.reg_write(UC_X86_REG_ECX, this)
        self.cpu.emu_start(function, 0x611000, count=200000)
        if self.cpu.reg_read(UC_X86_REG_EIP) != 0x611000:
            raise RuntimeError("original binary exceeded the instruction limit")
        return self.cpu.reg_read(UC_X86_REG_EAX)

    def entity(self, game_address):
        target = 0x602000
        # Define empty storage explicitly, as with the source reference's
        # repair for uninitialized object floats and the old partial HP read.
        self.cpu.mem_write(target, bytes(0x240))
        self.write(target, "2I", 0x600000, game_address)
        self.call(0x42B620, target, (1,))
        fields = {"x": (8, "f"), "y": (12, "f"), "xspeed": (16, "f"), "yspeed": (20, "f"),
                  "dir": (24, "b"), "act": (26, "h"), "frame": (28, "i"), "img": (32, "i"),
                  "hp": (36, "i"), "hitstop": (40, "h"), "fflags": (556, "I"), "aflags": (560, "I")}
        result = {name: self.read(target + offset, kind)[0] for name, (offset, kind) in fields.items()}
        for name, count_offset, box_offset in (("hitarea", 43, 44), ("attackarea", 42, 300)):
            count, = self.read(target + count_offset, "b")
            result[name] = tuple(self.read(target + box_offset + i * 16, "4i") for i in range(count))
        return result
