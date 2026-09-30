"""Read the pinned executable's Lua resource without loading or running it."""
import hashlib
from pathlib import Path
import struct


PACKAGE_SHA256 = "ff8fb443c227c0eeb0e1ff6f92ec0afed965d4f95e07f35247c3fb8579ab862f"


def embedded_api(executable):
    data = Path(executable).read_bytes()
    if hashlib.sha256(data).hexdigest() != PACKAGE_SHA256:
        raise ValueError("resource reference requires the exact audited executable")

    def integer(offset):
        return struct.unpack_from("<I", data, offset)[0]

    pe = integer(0x3C)
    optional = pe + 24
    section_count, = struct.unpack_from("<H", data, pe + 6)
    optional_size, = struct.unpack_from("<H", data, pe + 20)

    def file_offset(rva):
        for index in range(section_count):
            offset = optional + optional_size + index * 40
            _, address, size, raw = struct.unpack_from("<4I", data, offset + 8)
            if address <= rva < address + size:
                return raw + rva - address
        raise ValueError("resource address is outside the executable sections")

    root = file_offset(integer(optional + 96 + 2 * 8))

    def entries(relative):
        named, numbered = struct.unpack_from("<2H", data, root + relative + 12)
        result = []
        for index in range(named + numbered):
            name, child = struct.unpack_from("<2I", data, root + relative + 16 + index * 8)
            if name & 0x80000000:
                offset = root + (name & 0x7FFFFFFF)
                length, = struct.unpack_from("<H", data, offset)
                name = data[offset + 2:offset + 2 + length * 2].decode("utf-16le")
            result.append((name, child))
        return result

    directory = 0
    # The shipped executable uses the literal type name, not Windows type ID 10.
    for identifier in ("RT_RCDATA", 103):
        children = [child for name, child in entries(directory) if name == identifier]
        if len(children) != 1 or not children[0] & 0x80000000:
            raise ValueError("original Lua resource directory is missing or ambiguous")
        directory = children[0] & 0x7FFFFFFF
    languages = entries(directory)
    if len(languages) != 1 or languages[0][1] & 0x80000000:
        raise ValueError("original Lua resource language is missing or ambiguous")
    address, size = struct.unpack_from("<2I", data, root + languages[0][1])
    start = file_offset(address)
    return data[start:start + size]


def lua_program(bytecode):
    """Compare Lua 5.1 instructions and constants while excluding debug records."""
    if bytecode[:6] != b"\x1bLua\x51\0":
        raise ValueError("expected Lua 5.1 bytecode")
    endian, int_size, size_t, instruction_size, number_size, integral = bytecode[6:12]
    if endian not in (0, 1) or int_size != 4 or instruction_size != 4 or number_size != 8 or integral:
        raise ValueError("unsupported Lua 5.1 bytecode layout")
    order = "little" if endian else "big"
    cursor = 12

    def read(size):
        nonlocal cursor
        value = bytecode[cursor:cursor + size]
        if len(value) != size:
            raise ValueError("truncated Lua bytecode")
        cursor += size
        return value

    def integer():
        return int.from_bytes(read(int_size), order)

    def string():
        return read(int.from_bytes(read(size_t), order))

    def function():
        string()  # Source filename.
        read(2 * int_size)  # Source line range.
        arguments = read(4)  # Upvalues, arguments, varargs, stack size.
        instructions = read(integer() * instruction_size)
        constants = []
        for _ in range(integer()):
            kind = read(1)[0]
            if kind == 0:
                value = b""
            elif kind == 1:
                value = read(1)
            elif kind == 3:
                value = read(number_size)
            elif kind == 4:
                value = string()
            else:
                raise ValueError(f"invalid Lua constant type: {kind}")
            constants.append((kind, value))
        children = tuple(function() for _ in range(integer()))
        read(integer() * int_size)  # Instruction source lines.
        for _ in range(integer()):
            string()
            read(2 * int_size)  # Local variable scope.
        for _ in range(integer()):
            string()  # Upvalue debug names.
        return arguments, instructions, tuple(constants), children

    result = function()
    if cursor != len(bytecode):
        raise ValueError("unexpected Lua bytecode suffix")
    return result
