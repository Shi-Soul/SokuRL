#pragma once
#include <cstdint>

namespace SokuRLBridge {
#pragma pack(push, 4)
struct NetworkInputEvent {
    std::uint32_t request, kind, command, match, round, duration;
    std::uint64_t observed, target, at;
};
#pragma pack(pop)
static_assert(sizeof(NetworkInputEvent) == 48, "network input event layout");
// Kinds 1-7 are request results; 8 injected, 9 expired (legacy), 10 cancelled,
// 11 superseded by a newer intent while waiting for an original input window.
bool initializeNetworkInputEvents();
void appendNetworkInputEvent(const NetworkInputEvent &event);
void closeNetworkInputEvents();
}
