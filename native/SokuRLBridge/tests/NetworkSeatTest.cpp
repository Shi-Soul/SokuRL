#include "../NetworkState.hpp"
using namespace SokuRLBridge;
static_assert(networkSeat(UINT32_MAX, 8) == 0, "first host selection");
static_assert(networkSeat(UINT32_MAX, 9) == 1, "first client selection");
static_assert(networkSeat(1, 14) == 1, "first client battle");
static_assert(networkSeat(1, 8) == 1, "client victory dialogue enters shared selection");
static_assert(networkSeat(1, 10) == 1, "client loads rematch through shared scene");
static_assert(networkSeat(1, 13) == 1, "client keeps its seat in shared rematch battle");
static_assert(networkSeat(0, 13) == 0, "host keeps its seat");
static_assert(networkSeat(1, 2) == UINT32_MAX, "leaving netplay clears ownership");
int main() { return 0; }
