#pragma once
#include <cstdint>

namespace SokuRLBridge {
struct LocalStart {
    bool pauseAtStart = false;
    bool seedRequested = false;
    std::uint32_t seed = 0;
    std::uint32_t p1Character = 1;
    std::uint32_t p2Character = 0;
    std::uint32_t p1Palette = 0;
    std::uint32_t p2Palette = 0;
    std::uint32_t p1Deck = 0;
    std::uint32_t p2Deck = 0;
    std::uint32_t stage = 0;
    std::uint32_t music = 0;
};
std::uint32_t environmentValue(const wchar_t *name, std::uint32_t fallback);
LocalStart readLocalStart();
bool configureLocalStart(const LocalStart &config);
}
