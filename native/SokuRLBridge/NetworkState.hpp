#pragma once
#include "ControlBlock.hpp"
#include "RenderState.hpp"

namespace SokuRLBridge {
#pragma pack(push, 4)
struct NetworkState {
    std::uint32_t magic, version, size;
    volatile long sequence;
    std::uint32_t connected, scene, match, localSeat;
    std::uint64_t updates;
    std::uint32_t scores[2];
    RawFrameState raw;
    RenderState render;
};
#pragma pack(pop)
static_assert(offsetof(NetworkState, raw) == 48, "network header layout");
static_assert(sizeof(NetworkState) == 13272, "network state layout");

bool initializeNetworkState();
void closeNetworkState();
void observeNetworkScene(std::uint32_t scene);
std::uint64_t nextNetworkUpdate();
std::uint32_t networkMatch();
const NetworkState &currentNetworkState();
void publishNetworkState(const RawFrameState &raw, unsigned leftScore, unsigned rightScore);
}
