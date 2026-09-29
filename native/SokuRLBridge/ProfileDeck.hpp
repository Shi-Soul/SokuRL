#pragma once

#include <Profile.hpp>
#include <cstdint>

namespace SokuRLBridge {
inline SokuLib::Dequeue<unsigned short> &profileDeck(
    SokuLib::Profile &profile, std::uint32_t character, std::uint32_t deck)
{
    // th123 1.10a stores 20 * 4 deques at +0x1AC, ending at +0x7EC
    // (the distance between profile1 and profile2). The bundled SokuLib
    // Profile::cards member has an incorrect offset due to pointer padding.
    constexpr std::uintptr_t CARDS_OFFSET = 0x1AC;
    static_assert(sizeof(SokuLib::Dequeue<unsigned short>) == 20);
    auto *cards = reinterpret_cast<SokuLib::Dequeue<unsigned short> *>(
        reinterpret_cast<std::uintptr_t>(&profile) + CARDS_OFFSET);
    return cards[character * 4 + deck];
}
}
