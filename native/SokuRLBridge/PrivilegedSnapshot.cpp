#include "PrivilegedSnapshot.hpp"

namespace {
using namespace SokuRLBridge;
class Capture {
    PrivilegedSnapshot &out;
    SnapshotRead read;
public:
    Capture(PrivilegedSnapshot &snapshot, SnapshotRead reader) : out(snapshot), read(reader) {}
    bool valid(bool condition) {
        if (!condition && out.error == SnapshotError::None) out.error = SnapshotError::InvalidStructure;
        return condition && out.error == SnapshotError::None;
    }
    void add(std::uint32_t address, unsigned size) {
        if (out.error != SnapshotError::None) return;
        if (!valid(address && size && address <= UINT32_MAX-size)) return;
        if (out.regions == SNAPSHOT_REGIONS || size > SNAPSHOT_BYTES-out.bytes) {
            out.error = SnapshotError::Capacity;
            return;
        }
        if (!read(address, out.data+out.bytes, size)) {
            out.error = SnapshotError::ReadFailed;
            return;
        }
        out.index[out.regions++] = {address, size, out.bytes};
        out.bytes += size;
    }
    template<class T> T value(std::uint32_t address) {
        T value{};
        if (out.error == SnapshotError::None && !read(address, &value, sizeof(value)))
            out.error = SnapshotError::ReadFailed;
        return value;
    }
    void entity(std::uint32_t address, bool fighter) {
        add(address, fighter ? 0x930 : 0x360);
        add(value<std::uint32_t>(address+0x150), 0x54);
        const auto attacks = value<unsigned char>(address+0x1CB);
        if (!valid(attacks <= 16 && value<unsigned char>(address+0x1CC) <= 16)) return;
        for (unsigned i = 0; i < attacks; ++i) {
            const auto pointer = value<std::uint32_t>(address+0x320+i*4);
            if (pointer) add(pointer, 16);
        }
    }
    void deck(unsigned seat) {
        const auto start = 0x899D18+seat*0x20;
        const auto table = value<std::uint32_t>(start+4);
        const auto chunks = value<std::uint32_t>(start+8);
        const auto first = value<std::uint32_t>(start+12);
        if (!valid(chunks && value<std::uint32_t>(start+16) == 20)) return;
        for (unsigned i = 0; i < 20; ++i) {
            const auto slot = table+(((first+i)>>3)%chunks)*4;
            add(slot, 4);
            add(value<std::uint32_t>(slot)+((first+i)&7)*2, 2);
        }
    }
    void fighter(std::uint32_t address, unsigned seat, int weather) {
        entity(address, true);
        const auto table = value<std::uint32_t>(address+0x5EC);
        const auto maximum = value<std::uint32_t>(address+0x5F0);
        const auto point = value<std::uint32_t>(address+0x5F4);
        const auto count = value<std::int32_t>(address+0x5F8);
        if (!valid(count >= 0 && count <= 5)) return;
        const auto spell = value<std::uint16_t>(address+0x5E4) +
            value<unsigned char>(address+0x5E6)*500;
        for (unsigned i = 0; weather != 11 && maximum && i < 5 && i < spell/500; ++i) {
            const auto slot = table+((point+i)%maximum)*4;
            add(slot, 4);
            const auto card = value<std::uint32_t>(slot);
            if (card) add(card, 4);
        }
        if (weather != 11 && count > 0) {
            add(table+point*4, 4);
            add(value<std::uint32_t>(table+point*4), 4);
        }
        const auto keys = value<std::uint32_t>(address+0x750);
        add(keys, 4);
        add(value<std::uint32_t>(keys)+0x38, 32);
        deck(seat);
    }
    void objects(unsigned seat) {
        const auto root = value<std::uint32_t>(0x8985DC);
        add(root+0x40, 8);
        const auto head = value<std::uint32_t>(root+0x40);
        if (!valid(value<std::uint32_t>(root+0x44)-head == 8)) return;
        add(head, 8);
        const auto character = value<std::uint32_t>(head+seat*4);
        add(character+0x6F8, 4);
        const auto manager = value<std::uint32_t>(character+0x6F8);
        add(manager+0x58, 12);
        const auto sentinel = value<std::uint32_t>(manager+0x5C);
        const auto count = value<std::uint32_t>(manager+0x60);
        if (!valid(count <= 1024)) return;
        add(sentinel, 4);
        auto node = value<std::uint32_t>(sentinel);
        for (unsigned i = 0; i < count; ++i) {
            if (!valid(node && node != sentinel)) return;
            add(node, 12);
            entity(value<std::uint32_t>(node+8), false);
            node = value<std::uint32_t>(node);
        }
        valid(node == sentinel);
    }
};
}

namespace SokuRLBridge {
void capturePrivilegedSnapshot(PrivilegedSnapshot &snapshot, SnapshotRead read) {
    snapshot.regions = snapshot.bytes = 0;
    snapshot.error = SnapshotError::None;
    Capture capture(snapshot, read);
    capture.add(0x8971C0, 0x10);
    capture.add(0x8985D8, 0x10);
    capture.add(0x899D0D, 0x43);
    const auto battle = capture.value<std::uint32_t>(0x8985E4);
    capture.add(battle+0xC, 8);
    const auto weather = capture.value<std::int32_t>(0x8971C0);
    for (unsigned seat = 0; seat < 2; ++seat) {
        capture.fighter(capture.value<std::uint32_t>(battle+0xC+seat*4), seat, weather);
        capture.objects(seat);
    }
}
}
