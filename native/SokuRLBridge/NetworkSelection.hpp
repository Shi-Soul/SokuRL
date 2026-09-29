#pragma once

namespace SokuRLBridge {
// th123 1.10a's character order, verified against the original executable.
// The final slot is random. Inputs still pass through the original menu.
constexpr unsigned SELECTION_CHARACTERS[] = {
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 20
};
constexpr unsigned characterCursor(unsigned character) {
    for (unsigned cursor = 0; cursor < 21; ++cursor)
        if (SELECTION_CHARACTERS[cursor] == character) return cursor;
    return 21;
}

constexpr int selectionDirection(unsigned cursor, unsigned target) {
    const auto forward = (target + 21 - cursor) % 21;
    return forward == 0 ? 0 : forward <= 10 ? 1 : -1;
}
}
