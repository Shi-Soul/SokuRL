#include "../FrameState.hpp"
#include <cstdio>

int main() {
    SokuRLBridge::RawFrameState state{};
    // Constants were computed by the independent Python bridge serializer.
    if (SokuRLBridge::stateHash(state) != 0xC565BC7649871C05ULL) return 1;
    state.frameId = 519;
    state.sceneId = 5;
    state.battleMode = 3;
    state.p1.spirit = -88;
    state.p1.x = 123.5f;
    state.p1.input.horizontalAxis = -12;
    state.p2.hp = 9999;
    state.p2.handIds[4] = -1;
    state.p1ObjectCount = 1;
    state.p1Objects[0].typeId = 0x12345678;
    state.p1Objects[0].speedY = -3.25f;
    const auto actual = SokuRLBridge::stateHash(state);
    if (actual != 0x200CF9085327CE3CULL) {
        std::printf("frame hash differs from Python: %llx\n", actual);
        return 2;
    }
    return 0;
}
