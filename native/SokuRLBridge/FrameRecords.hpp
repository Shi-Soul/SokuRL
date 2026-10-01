#pragma once
#include "ControlBlock.hpp"
#include <cstddef>

namespace SokuRLBridge {
std::uint32_t load32(const volatile std::uint32_t *value);
void store32(volatile std::uint32_t *target, std::uint32_t value);
void beginStatusWrite(ControlBlock *control);
void endStatusWrite(ControlBlock *control);

class FrameRecords {
public:
    explicit FrameRecords(BridgeMapping &mapping);
    void clear();
    void recordInitial(const RawFrameState &state);
    void trim(std::uint64_t frame);
    void invalidateCheckpoint(ResultCode reason);
    void publishLatest(const RawFrameState &state, std::uint32_t remaining);
    void pushRing(const RawFrameState &state);
    void appendRecordedFrame(const RawFrameState &state, std::uint32_t remaining);
private:
    void publishReconstructionFrame(const RawFrameState &state);
    BridgeMapping &mapping_;
    ControlBlock &control_;
    std::size_t count_ = 0;
};
}
