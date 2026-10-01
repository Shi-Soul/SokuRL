#pragma once
#include "RealtimeInputSchedule.hpp"

namespace SokuRLBridge {
class ControlledInput;
#pragma pack(push, 4)
struct RealtimeInputBlock {
    std::uint32_t magic, version, size, seat;
    volatile long requestGuard;
    RealtimeCommand command;
    volatile long statusGuard;
    std::uint32_t acknowledged, result, match, round;
    std::uint64_t frame;
    std::uint32_t appliedSequence;
    std::uint64_t appliedAt;
    std::uint32_t pending;
    LogicalInput held;
};
#pragma pack(pop)
static_assert(sizeof(RealtimeCommand) == 72, "realtime command layout");
static_assert(sizeof(RealtimeInputBlock) == 168, "realtime input layout");

bool initializeRealtimeInput();
void closeRealtimeInput();
bool realtimeInputEnabled();
void prepareRealtimeInput(ControlledInput &inputs, std::uint32_t match,
    std::uint32_t round, std::uint64_t frame);
}
