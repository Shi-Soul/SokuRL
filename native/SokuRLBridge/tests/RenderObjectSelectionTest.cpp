#include "../RenderState.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
using namespace SokuRLBridge;
void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
int main() {
    RenderState state{};
    state.cameraY = 480;
    state.cameraScale = 1;
    // The live failure had 67 engine objects, but at most 30 in the viewport.
    // Engine list length must not exhaust the public renderer's capacity.
    for (unsigned i = 0; i < 40; ++i) captureRenderObject(state, {-32, 100, 1, 1, 1}, 1);
    for (unsigned i = 0; i < 27; ++i) captureRenderObject(state, {float(i), 100, 1, 1, 1}, 1);
    require(!state.overflow && state.counts[1] == 27, "invisible objects consumed render capacity");
    for (unsigned i = 0; i < 100; ++i) {
        captureRenderObject(state, {10, 100, 0, 1, 1}, 1);
        captureRenderObject(state, {10, 100, 1, 1, 0}, 1);
    }
    require(state.counts[1] == 27 && !state.overflow, "transparent or undrawable objects consumed capacity");
    for (unsigned i = 27; i < 65; ++i) captureRenderObject(state, {float(i), 100, 1, 1, 1}, 1);
    require(state.overflow && state.counts[1] == 64, "real visible overflow was hidden");
    captureRenderObject(state, {std::numeric_limits<float>::quiet_NaN(), 100, 1, 1, 1}, 0);
    require(state.counts[0] == 1 && std::isnan(state.objects[0][0].x), "invalid geometry was silently dropped");
}
