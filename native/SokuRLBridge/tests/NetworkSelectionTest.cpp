#include "../NetworkSelection.hpp"
#include <cassert>

int main() {
    using namespace SokuRLBridge;
    static_assert(characterCursor(0) == 9 && characterCursor(1) == 8);
    static_assert(selectionDirection(9, 8) == -1);
    static_assert(selectionDirection(8, 9) == 1);
    static_assert(selectionDirection(20, 8) == 1);
    static_assert(selectionDirection(0, 9) == 1);
    for (unsigned character = 0; character < 2; ++character) {
        const auto target = characterCursor(character);
        for (unsigned initial = 0; initial <= 20; ++initial) {
            auto cursor = initial;
            unsigned steps = 0;
            while (cursor != target && steps <= 10) {
                cursor = (cursor + 21 + selectionDirection(cursor, target)) % 21;
                ++steps;
            }
            assert(cursor == target && steps <= 10);
            assert(selectionDirection(cursor, target) == 0);
        }
    }
}
