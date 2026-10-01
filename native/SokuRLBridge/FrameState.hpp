#pragma once
#include "ControlBlock.hpp"

namespace SokuLib { struct BattleManager; }

namespace SokuRLBridge {
struct CheckpointIdentity {
    std::uint32_t leftCharacter;
    std::uint32_t rightCharacter;
    std::uint32_t stage;
    std::uint32_t randomSeed;
    std::uint32_t practiceWeather;
    std::uint32_t dummyState;
    std::uint32_t position;
    std::uint32_t guard;
    std::uint32_t counter;
    std::uint32_t airtech;
};

CheckpointIdentity readCheckpointIdentity();
bool identityMatchesCheckpoint(const CheckpointIdentity &checkpoint);
bool isPracticeGameplay();
bool isReplayGameplay();
bool isLocalVersusGameplay();
bool isSupportedGameplay();

SimpleStatePatch simplePatchFrom(const RawFrameState &state);
bool isValidSimplePlayerState(const SimplePlayerState &state);
void applySimpleState(SokuLib::BattleManager &manager, const SimpleStatePatch &state);
std::uint64_t stateHash(const RawFrameState &state);
RawFrameState captureState(SokuLib::BattleManager *manager, std::uint64_t frame,
    std::uint32_t segment, const LogicalInput (&inputs)[2]);
}
