#pragma once
#include "ControlBlock.hpp"
#include "PrivilegedSnapshot.hpp"
#include "RenderState.hpp"

namespace SokuRLBridge {
constexpr unsigned REALTIME_HISTORY_CAPACITY = 16;
#pragma pack(push, 4)
struct RealtimeFrame {
    volatile long guard;
    std::uint32_t match;
    std::uint64_t frame;
    std::uint32_t scores[2];
    std::uint64_t captureTicks;
    RawFrameState raw;
    RenderState render;
    PrivilegedSnapshot memory;
};
struct RealtimeHistory {
    std::uint32_t magic, version, size, capacity;
    std::uint64_t frequency;
    volatile long published;
    std::uint32_t alive;
    RealtimeFrame frames[REALTIME_HISTORY_CAPACITY];
};
#pragma pack(pop)
bool initializeRealtimeObservation();
bool initializeOfflineObservation();
void closeRealtimeObservation();
void publishRealtimeObservation(const RawFrameState &raw);
}
