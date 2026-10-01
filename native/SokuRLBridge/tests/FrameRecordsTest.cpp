#include "../FrameRecords.hpp"
#include <memory>
#include <stdexcept>

using namespace SokuRLBridge;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    auto mapping = std::make_unique<BridgeMapping>();
    auto &control = mapping->control;
    FrameRecords records(*mapping);
    RawFrameState state{};
    state.p1.spirit = -88;
    state.p2.input.a = 3;
    state.stateHash = 123;
    records.recordInitial(state);
    records.publishLatest(state, 7);
    records.pushRing(state);
    require(control.checkpointValid == 1 && control.recordedFrames == 1, "initial history");
    require(control.statusSeq == 2 && control.stepsRemaining == 7, "status publication");
    require(mapping->history[0].simple.p1.spirit == -88 &&
        mapping->history[0].p2Input.a == 3 && mapping->history[0].stateHash == 123, "input history payload");
    for (std::uint64_t frame = 1; frame < INPUT_HISTORY_CAPACITY; ++frame) {
        control.ringReadSeq = control.ringWriteSeq;
        state.frameId = frame;
        records.appendRecordedFrame(state, 1);
    }
    require(control.recordedFrames == INPUT_HISTORY_CAPACITY && control.checkpointValid == 1,
        "last valid history frame");
    state.frameId = INPUT_HISTORY_CAPACITY;
    records.appendRecordedFrame(state, 0);
    require(control.checkpointValid == 0 && control.recordedFrames == 0 &&
        control.resultCode == static_cast<unsigned>(ResultCode::HistoryFull), "history overflow");
    require(control.latest.frameId == INPUT_HISTORY_CAPACITY, "publish after history overflow");
    state.frameId = 0;
    records.recordInitial(state);
    state.frameId = 3;
    records.appendRecordedFrame(state, 0);
    require(control.checkpointValid == 0 && control.recordedFrames == 0 &&
        control.resultCode == static_cast<unsigned>(ResultCode::CheckpointInvalidated), "history gap");
    control.ringReadSeq = control.ringWriteSeq = 0xFFFFFFFFU;
    records.pushRing(state);
    require(control.ringWriteSeq == 0 && mapping->frames[FRAME_RING_CAPACITY - 1].frameId == 3,
        "ring sequence wrap");
    control.ringReadSeq = control.ringWriteSeq - FRAME_RING_CAPACITY;
    const auto dropped = control.droppedFrames;
    records.pushRing(state);
    require(control.ringWriteSeq == 0 && control.droppedFrames == dropped + 1, "ring full");
    return 0;
}
