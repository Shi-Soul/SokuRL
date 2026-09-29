#pragma once
#include "ControlBlock.hpp"
#include <deque>

namespace SokuRLBridge {
struct ScheduledNetworkInput {
    unsigned sequence, command, match, round, duration;
    std::uint64_t observed, target;
    LogicalInput input;
};

class NetworkInputSchedule {
    using Recorder = void (*)(const ScheduledNetworkInput &, unsigned, std::uint64_t);
    Recorder record;
    std::deque<ScheduledNetworkInput> pending;
    LogicalInput held{};
    std::uint64_t expires = 0, nextInjection = 0;
public:
    explicit NetworkInputSchedule(Recorder recorder) : record(recorder) {}
    std::size_t size() const { return pending.size(); }
    void push(const ScheduledNetworkInput &request) { pending.push_back(request); }
    void clear(std::uint64_t at) {
        for (const auto &request : pending) record(request, 10, at);
        pending.clear();
        held = {};
        expires = 0;
        nextInjection = 0;
    }
    const LogicalInput &apply(std::uint64_t at) {
        if (at >= expires) held = {};
        if (at < nextInjection || pending.empty() || pending.front().target > at) return held;
        // Original netplay can advance several battle updates between input
        // windows. Apply the latest due intent once, never replay a burst.
        while (pending.size() > 1 && pending[1].target <= at) {
            record(pending.front(), 11, at);
            pending.pop_front();
        }
        const auto request = pending.front();
        pending.pop_front();
        held = request.input;
        expires = at + request.duration;
        nextInjection = at + 3;
        record(request, 8, at);
        return held;
    }
};
}
