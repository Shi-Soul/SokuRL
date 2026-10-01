#pragma once
#include "ControlBlock.hpp"

namespace SokuRLBridge {
using InputDecoder = LogicalInput (*)(const LogicalInput &, const LogicalInput &);
bool isValidInput(const LogicalInput &input, std::uint32_t duration);

class ControlledInput {
public:
    explicit ControlledInput(InputDecoder decoder);
    void request(const LogicalInput &p1, const LogicalInput &p2, std::uint32_t mask, std::uint32_t duration);
    void clear(bool neutral);
    void clearEffective();
    void resetHistory();
    void resetEpisode();
    LogicalInput apply(unsigned player, const LogicalInput &physical,
        bool simulating, bool battleActive, bool paused);
    void finishFrame(ControlBlock &control);
    LogicalInput effective[2]{};
private:
    InputDecoder decoder_;
    LogicalInput simulated_[2]{};
    LogicalInput requested_[2]{};
    std::uint32_t mask_ = 0;
    std::uint32_t remaining_ = 0;
    bool neutralPending_ = false;
};
}
