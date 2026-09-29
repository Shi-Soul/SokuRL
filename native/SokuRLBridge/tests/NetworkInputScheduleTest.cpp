#include "../NetworkInputSchedule.hpp"
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

using namespace SokuRLBridge;
using Event = std::tuple<unsigned, unsigned, std::uint64_t>;
std::vector<Event> events;
void record(const ScheduledNetworkInput &request, unsigned kind, std::uint64_t at) {
    events.emplace_back(request.sequence, kind, at);
}
void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
ScheduledNetworkInput request(unsigned sequence, std::uint64_t observed, int horizontal) {
    return {sequence, 1, 1, 0, 3, observed, observed+5, {horizontal, 0, 0, 0, 0, 0, 0, 0}};
}

int main() {
    NetworkInputSchedule schedule(record);
    schedule.push(request(69, 205, 1));
    require(schedule.apply(209).horizontalAxis == 0, "input arrived before latency");
    require(schedule.apply(210).horizontalAxis == 1, "exact boundary was missed");
    schedule.push(request(70, 208, -1));
    schedule.push(request(71, 211, 1));
    require(schedule.apply(212).horizontalAxis == 1, "hold duration was shortened");
    // Real client trace: the input hook is absent at 213 and next runs at 214.
    require(schedule.apply(214).horizontalAxis == -1, "network update skipped the input boundary");
    require(schedule.apply(216).horizontalAxis == -1, "catch-up exceeded the control frequency");
    require(schedule.apply(217).horizontalAxis == 1, "next available input window was missed");
    require(events == std::vector<Event>{{69, 8, 210}, {70, 8, 214}, {71, 8, 217}}, "incorrect timing history");
    require(schedule.apply(220).horizontalAxis == 0, "expired hold was not released");

    events.clear();
    schedule.clear(221);
    schedule.push(request(72, 221, -1));
    schedule.push(request(73, 224, 1));
    require(schedule.apply(230).horizontalAxis == 1, "catch-up did not choose the latest due command");
    require(events == std::vector<Event>{{72, 11, 230}, {73, 8, 230}}, "superseded command was not recorded");
    schedule.push(request(74, 227, -1));
    schedule.clear(231);
    require(events.back() == Event{74, 10, 231}, "round reset did not cancel pending input");
    require(schedule.apply(232).horizontalAxis == 0 && schedule.size() == 0, "round reset leaked input");
}
