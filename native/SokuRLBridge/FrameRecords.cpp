#include "FrameRecords.hpp"
#include "FrameState.hpp"
#include <Windows.h>

namespace SokuRLBridge {
std::uint32_t load32(const volatile std::uint32_t *value)
{
    return static_cast<std::uint32_t>(InterlockedCompareExchange(
        reinterpret_cast<volatile LONG *>(const_cast<volatile std::uint32_t *>(value)), 0, 0));
}

void store32(volatile std::uint32_t *target, std::uint32_t value)
{
    InterlockedExchange(reinterpret_cast<volatile LONG *>(target), static_cast<LONG>(value));
}

void beginStatusWrite(ControlBlock *control)
{
    InterlockedIncrement(reinterpret_cast<volatile LONG *>(&control->statusSeq));
    MemoryBarrier();
}

void endStatusWrite(ControlBlock *control)
{
    MemoryBarrier();
    InterlockedIncrement(reinterpret_cast<volatile LONG *>(&control->statusSeq));
}

FrameRecords::FrameRecords(BridgeMapping &mapping) : mapping_(mapping), control_(mapping.control) {}

void FrameRecords::clear() { count_ = 0; }

void FrameRecords::recordInitial(const RawFrameState &state)
{
    count_ = 1;
    publishReconstructionFrame(state);
    store32(&control_.checkpointValid, 1);
}

void FrameRecords::trim(std::uint64_t frame)
{
    if (load32(&control_.checkpointValid) && count_ > frame + 1) {
        count_ = static_cast<std::size_t>(frame + 1);
        store32(&control_.validationState, static_cast<std::uint32_t>(ValidationState::Unknown));
    }
}

void FrameRecords::invalidateCheckpoint(SokuRLBridge::ResultCode reason)
{
    if (!load32(&control_.checkpointValid))
        return;
    store32(&control_.checkpointValid, 0);
    count_ = 0;
    store32(&control_.resultCode, static_cast<std::uint32_t>(reason));
}

void FrameRecords::publishReconstructionFrame(const SokuRLBridge::RawFrameState &state)
{
    if (state.frameId >= SokuRLBridge::INPUT_HISTORY_CAPACITY)
        return;
    auto &target = mapping_.history[state.frameId];
    target.p1Input = state.p1.input;
    target.p2Input = state.p2.input;
    target.simple = simplePatchFrom(state);
    target.stateHash = state.stateHash;
}

void FrameRecords::publishLatest(const RawFrameState &state, std::uint32_t remaining)
{
    beginStatusWrite(&control_);
    control_.currentFrame = state.frameId;
    control_.latest = state;
    control_.recordedFrames = count_;
    control_.stepsRemaining = remaining;
    endStatusWrite(&control_);
}

void FrameRecords::pushRing(const SokuRLBridge::RawFrameState &state)
{
    const auto write = load32(&control_.ringWriteSeq);
    const auto read = load32(&control_.ringReadSeq);
    if (write - read >= SokuRLBridge::FRAME_RING_CAPACITY) {
        InterlockedIncrement(reinterpret_cast<volatile LONG *>(&control_.droppedFrames));
        return;
    }
    mapping_.frames[write % SokuRLBridge::FRAME_RING_CAPACITY] = state;
    MemoryBarrier();
    store32(&control_.ringWriteSeq, write + 1);
}

void FrameRecords::appendRecordedFrame(const RawFrameState &state, std::uint32_t remaining)
{
    if (load32(&control_.checkpointValid)) {
        if (count_ != state.frameId) {
            invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
        } else if (count_ < SokuRLBridge::INPUT_HISTORY_CAPACITY) {
            ++count_;
            publishReconstructionFrame(state);
        } else {
            invalidateCheckpoint(SokuRLBridge::ResultCode::HistoryFull);
        }
    }
    publishLatest(state, remaining);
    pushRing(state);
}

}
