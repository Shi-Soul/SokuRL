#pragma once
#include "ControlBlock.hpp"

namespace SokuLib { struct BattleManager; }

namespace SokuRLBridge {
std::uint64_t stateHash(const RawFrameState &state);
RawFrameState captureState(SokuLib::BattleManager *manager, std::uint64_t frame,
    std::uint32_t segment, const LogicalInput (&inputs)[2]);
}
