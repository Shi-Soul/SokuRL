"""Read every retained netplay update; never hide a consumer history overrun."""
import ctypes
import struct
import time

import bridge_shared
from network_runtime.state import MAPPING_SIZE, decode_network_state


HEADER = struct.Struct("<6IQ")
CAPACITY = 256
MAGIC = 0x484E4B53
SIZE = HEADER.size + CAPACITY * MAPPING_SIZE


def retained_indices(written, cursor):
    if type(cursor) is not int or cursor < 0 or cursor > written:
        raise ValueError("cursor must identify a previously read history entry")
    if written - cursor > CAPACITY:
        raise BufferError(f"network history lost {written - cursor - CAPACITY} entries")
    return tuple(index % CAPACITY for index in range(cursor, written))


class NetworkHistoryReader:
    """Copy a typed network history without changing its producer or cursor."""

    def __init__(self, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("a positive game PID is required")
        kernel = bridge_shared._kernel32
        self.handle = kernel.OpenFileMappingW(4, False, rf"Local\SokuRL{self.mapping}_{pid}")
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "network history is unavailable")
        size = HEADER.size + CAPACITY*self.entry_size
        self.view = kernel.MapViewOfFile(self.handle, 4, 0, 0, size)
        if not self.view:
            error = ctypes.get_last_error()
            kernel.CloseHandle(self.handle)
            self.handle = None
            raise OSError(error, "cannot map network history")
        if struct.unpack("<4I", ctypes.string_at(self.view, 16)) != (self.magic, self.version, size, CAPACITY):
            self.close()
            raise ValueError("unsupported network history ABI")

    def read_after(self, cursor, timeout):
        if not self.view or timeout <= 0:
            raise ValueError("history read requires an open mapping and positive timeout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            before = ctypes.c_uint32.from_address(self.view + 16).value
            if before & 1:
                time.sleep(.001)
                continue
            header = HEADER.unpack(ctypes.string_at(self.view, HEADER.size))
            written = header[6]
            # Defer validation until the complete copy has a stable sequence.
            valid_range = type(cursor) is int and 0 <= cursor <= written and written-cursor <= CAPACITY
            indices = retained_indices(written, cursor) if valid_range else ()
            entries = [ctypes.string_at(self.view + HEADER.size + index*self.entry_size, self.entry_size)
                       for index in indices]
            after = ctypes.c_uint32.from_address(self.view + 16).value
            if before != after or before != header[4]:
                continue
            if not header[5]:
                raise EOFError("network history producer closed")
            retained_indices(written, cursor)
            return written, tuple(self.decode(entry) for entry in entries)
        raise TimeoutError("could not copy stable network history")

    def close(self):
        kernel = bridge_shared._kernel32
        if self.view:
            kernel.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            kernel.CloseHandle(self.handle)
            self.handle = None


class NetworkHistoryClient(NetworkHistoryReader):
    """A cursor counts records, including scene transitions, not game frames.

    Start with cursor 0 before entering the match. Each read returns the next
    cursor and ordered snapshots. An empty batch means no new record exists.
    A closed producer raises EOFError; overwritten records raise BufferError.
    The reader cannot pause the game or advance the writer's position.
    """
    mapping = "NetworkHistory"
    version = 2
    magic = MAGIC
    entry_size = MAPPING_SIZE
    decode = staticmethod(decode_network_state)
