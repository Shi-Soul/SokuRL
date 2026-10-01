#pragma once
#include "ControlBlock.hpp"

namespace SokuLib { struct BattleManager; }

namespace SokuRLBridge {
SimpleStatePatch simplePatchFrom(const RawFrameState &state);
bool isValidSimplePlayerState(const SimplePlayerState &state);
void applySimpleState(SokuLib::BattleManager &manager, const SimpleStatePatch &state);
std::uint64_t stateHash(const RawFrameState &state);
RawFrameState captureState(SokuLib::BattleManager *manager, std::uint64_t frame,
    std::uint32_t segment, const LogicalInput (&inputs)[2]);
}
