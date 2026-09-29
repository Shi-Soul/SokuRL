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
    std::uint64_t expires = 0;
public:
    explicit NetworkInputSchedule(Recorder recorder) : record(recorder) {}
    std::size_t size() const { return pending.size(); }
    void push(const ScheduledNetworkInput &request) { pending.push_back(request); }
    void clear(std::uint64_t at) {
        for (const auto &request : pending) record(request, 10, at);
        pending.clear();
        held = {};
        expires = 0;
    }
    const LogicalInput &apply(std::uint64_t at) {
        if (at >= expires) held = {};
        while (!pending.empty() && pending.front().target <= at) {
            const auto request = pending.front();
            pending.pop_front();
            if (request.target != at) { record(request, 9, at); continue; }
            held = request.input;
            expires = request.target + request.duration;
            record(request, 8, at);
        }
        return held;
    }
};
}
