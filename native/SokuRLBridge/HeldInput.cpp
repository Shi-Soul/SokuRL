#include "HeldInput.hpp"
#include <cstddef>

namespace SokuRLBridge {
namespace {
struct ReplayKeymap {
    unsigned char prefix[0x38];
    LogicalInput input;
    int pause;
    int select;
    unsigned short packed;
    unsigned short previousPacked;
    unsigned char replayEnabled;
    unsigned char padding[3];
};
static_assert(offsetof(ReplayKeymap, input) == 0x38);
static_assert(offsetof(ReplayKeymap, packed) == 0x60);
static_assert(offsetof(ReplayKeymap, replayEnabled) == 0x64);
}

LogicalInput advanceHeldInput(const LogicalInput &previous, const LogicalInput &intent) {
    ReplayKeymap state{};
    state.input = previous;
    state.packed = static_cast<unsigned short>(
        (intent.verticalAxis < 0 ? 1 : intent.verticalAxis > 0 ? 2 : 0) |
        (intent.horizontalAxis < 0 ? 4 : intent.horizontalAxis > 0 ? 8 : 0) |
        (intent.a << 4) | (intent.b << 5) | (intent.c << 6) | (intent.d << 7) |
        (intent.changeCard << 8) | (intent.spellcard << 9));
    state.replayEnabled = 1;
    // This branch reads only the supplied keymap. It increments held counters,
    // resets released keys and starts changed directions at +/-1 exactly as
    // the original game does. It does not call the physical keyboard hook.
    reinterpret_cast<void (__thiscall *)(ReplayKeymap *)>(0x0040A370)(&state);
    return state.input;
}
}
