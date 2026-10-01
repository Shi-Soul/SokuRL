#pragma once
#include <cstdint>

namespace SokuRLBridge {
constexpr unsigned SNAPSHOT_REGIONS = 40000;
constexpr unsigned SNAPSHOT_BYTES = 4 * 1024 * 1024;
using SnapshotRead = bool (*)(std::uint32_t, void *, std::uint32_t);
enum class SnapshotError : std::uint32_t { None, ReadFailed, Capacity, InvalidStructure };
#pragma pack(push, 4)
struct MemoryRegion { std::uint32_t address, size, offset; };
struct PrivilegedSnapshot {
    std::uint32_t regions, bytes;
    SnapshotError error;
    MemoryRegion index[SNAPSHOT_REGIONS];
    unsigned char data[SNAPSHOT_BYTES];
};
#pragma pack(pop)
// Copy all memory read by PrivilegedReader at one simulation boundary. The
// callback never waits for a policy; errors stop capture, not game simulation.
void capturePrivilegedSnapshot(PrivilegedSnapshot &snapshot, SnapshotRead read);
}
