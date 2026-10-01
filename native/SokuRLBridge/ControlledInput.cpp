#include "ControlledInput.hpp"
#include "FrameRecords.hpp"

namespace SokuRLBridge {
namespace {
bool isBoolean(std::int32_t value) { return value == 0 || value == 1; }
}

bool isValidInput(const LogicalInput &input, std::uint32_t duration)
{
    return input.horizontalAxis >= -1 && input.horizontalAxis <= 1 &&
        input.verticalAxis >= -1 && input.verticalAxis <= 1 &&
        isBoolean(input.a) && isBoolean(input.b) && isBoolean(input.c) && isBoolean(input.d) &&
        isBoolean(input.changeCard) && isBoolean(input.spellcard) &&
        duration >= 1 && duration <= MAX_DURATION_FRAMES;
}

ControlledInput::ControlledInput(InputDecoder decoder) : decoder_(decoder) {}

void ControlledInput::request(const LogicalInput &p1, const LogicalInput &p2,
    std::uint32_t mask, std::uint32_t duration)
{
    requested_[0] = p1;
    requested_[1] = p2;
    mask_ = mask;
    remaining_ = duration;
    neutralPending_ = false;
}

void ControlledInput::clear(bool neutral)
{
    requested_[0] = {};
    requested_[1] = {};
    mask_ = remaining_ = 0;
    neutralPending_ = neutral;
}

void ControlledInput::clearEffective() { effective[0] = {}; effective[1] = {}; }

void ControlledInput::resetHistory()
{
    clearEffective();
    simulated_[0] = {};
    simulated_[1] = {};
}

void ControlledInput::resetEpisode()
{
    clearEffective();
    neutralPending_ = false;
}

LogicalInput ControlledInput::apply(unsigned player, const LogicalInput &physical,
    bool simulating, bool battleActive, bool paused)
{
    if (!simulating)
        return battleActive && paused ? simulated_[player] : physical;
    auto input = physical;
    if ((mask_ & (1U << player)) && remaining_) {
        simulated_[player] = decoder_(simulated_[player], requested_[player]);
        input = simulated_[player];
    } else if (player == 0 && neutralPending_) {
        input = {};
        neutralPending_ = false;
    }
    effective[player] = input;
    simulated_[player] = input;
    return input;
}

void ControlledInput::finishFrame(ControlBlock &control)
{
    if (!remaining_) return;
    --remaining_;
    store32(&control.inputFramesRemaining, remaining_);
    if (!remaining_) {
        mask_ = 0;
        neutralPending_ = true;
        store32(&control.resultCode, static_cast<std::uint32_t>(ResultCode::Complete));
    }
}
}
