#pragma once
#include "ControlBlock.hpp"
#include "RenderState.hpp"

namespace SokuRLBridge {
constexpr unsigned networkSeat(unsigned previous, unsigned scene) {
    const bool hostScene = scene == 8 || scene == 10 || scene == 13;
    const bool clientScene = scene == 9 || scene == 11 || scene == 14;
    if (!hostScene && !clientScene) return UINT32_MAX;
    // Original BattleClient::onProcess returns scene 8 after local victory
    // dialogue (0x4286C4). That shared scene does not change connection ownership.
    return previous <= 1 ? previous : clientScene ? 1 : 0;
}
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
static_assert(sizeof(NetworkState) == 51672, "network state layout");

bool initializeNetworkState();
void closeNetworkState();
void observeNetworkScene(std::uint32_t scene);
std::uint64_t nextNetworkUpdate();
std::uint32_t networkMatch();
const NetworkState &currentNetworkState();
void publishNetworkState(const RawFrameState &raw, unsigned leftScore, unsigned rightScore);
}
