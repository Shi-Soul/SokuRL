#pragma once

namespace SokuRLBridge {
// th123 1.10a's 21-entry character-select order starts 9,8,...,1,0.
// Character IDs are not cursor positions: Marisa is cursor 8, Reimu cursor 9.
constexpr unsigned characterCursor(unsigned character) { return 9 - character; }

constexpr int selectionDirection(unsigned cursor, unsigned target) {
    const auto forward = (target + 21 - cursor) % 21;
    return forward == 0 ? 0 : forward <= 10 ? 1 : -1;
}
}
