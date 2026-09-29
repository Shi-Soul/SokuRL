#pragma once
#include "ControlBlock.hpp"
namespace SokuLib { struct KeymapManager; }
namespace SokuRLBridge {
#pragma pack(push, 4)
struct NetworkInputBlock {
    std::uint32_t magic, version, size, requestSequence;
    std::uint32_t ackSequence, result, command, match, round;
    std::uint32_t duration;
    std::uint64_t observed, target;
    LogicalInput input;
    volatile long statusSequence;
    std::uint32_t injectedSequence;
    std::uint64_t injectedAt;
};
#pragma pack(pop)
static_assert(sizeof(NetworkInputBlock) == 104, "network input layout");
bool initializeNetworkInput();
void closeNetworkInput();
void serviceNetworkInput();
void applyNetworkInput(SokuLib::KeymapManager *keyboard);
}
