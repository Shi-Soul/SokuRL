#pragma once
#include <cstdint>
#include <cmath>

namespace SokuRLBridge {
constexpr unsigned RENDER_OBJECTS_PER_PLAYER = 64;
#pragma pack(push, 4)
struct RenderEntity {
    float x;
    float y;
    float alpha;
    std::int32_t facing;
    std::uint32_t drawable;
};

struct RenderState {
    float cameraX;
    float cameraY;
    float cameraScale;
    std::uint32_t weather;
    RenderEntity players[2];
    RenderEntity objects[2][RENDER_OBJECTS_PER_PLAYER];
    std::uint32_t counts[2];
    std::uint32_t overflow;
};
#pragma pack(pop)

inline void captureRenderObject(RenderState &state, const RenderEntity &object, unsigned owner) {
    // Match Python's projection in double precision. Remove only objects which
    // cannot pass any valid visibility configuration; keep invalid data visible
    // to the Python validator instead of hiding it through a comparison with NaN.
    const double x = (double(object.x) + state.cameraX) * state.cameraScale;
    const double y = (double(state.cameraY) - object.y) * state.cameraScale;
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(object.alpha) && state.cameraScale > 0 &&
        (object.drawable == 0 || object.alpha == 0 || x < 0 || x >= 640 || y < 0 || y >= 480)) return;
    if (state.counts[owner] == RENDER_OBJECTS_PER_PLAYER) { state.overflow = 1; return; }
    state.objects[owner][state.counts[owner]++] = object;
}

void captureRenderState(RenderState &state);
}
