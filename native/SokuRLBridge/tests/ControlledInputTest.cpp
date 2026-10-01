#include "../ControlledInput.hpp"
#include <stdexcept>

using namespace SokuRLBridge;
unsigned calls = 0;
LogicalInput decode(const LogicalInput &previous, const LogicalInput &intent)
{
    ++calls;
    auto result = intent;
    result.a = intent.a ? previous.a + 1 : 0;
    return result;
}
void require(bool value, const char *message)
{
    if (!value) throw std::runtime_error(message);
}
int main()
{
    ControlledInput inputs(decode);
    ControlBlock control{};
    LogicalInput pressed{};
    pressed.a = 1;
    LogicalInput physical{};
    physical.b = 7;
    inputs.request({}, pressed, 2, 2);
    require(inputs.apply(0, physical, true, true, true).b == 7, "uncontrolled seat keeps physical input");
    require(inputs.apply(1, physical, true, true, true).a == 1 && calls == 1, "controlled seat decodes once");
    inputs.finishFrame(control);
    require(control.inputFramesRemaining == 1, "first simulation consumes one frame");
    for (int poll = 0; poll < 100; ++poll)
        require(inputs.apply(1, physical, false, true, true).a == 1, "pause retains simulated input");
    require(calls == 1 && control.inputFramesRemaining == 1, "paused polling does not consume action");
    require(inputs.apply(1, physical, true, true, true).a == 2 && calls == 2, "next simulation advances held count");
    inputs.finishFrame(control);
    require(control.inputFramesRemaining == 0 &&
        control.resultCode == static_cast<unsigned>(ResultCode::Complete), "action completes at its duration");
    require(inputs.apply(1, physical, true, true, false).b == 7, "expired seat returns to physical input");
    require(inputs.apply(0, physical, true, true, false).b == 0, "legacy release clears P1 once");
    require(inputs.apply(0, physical, true, true, false).b == 7, "release is consumed once");
    inputs.clear(true);
    inputs.request(pressed, {}, 1, 1);
    require(inputs.apply(0, physical, true, true, true).a == 1, "new request replaces pending release");
    require(inputs.apply(1, physical, false, false, true).b == 7, "inactive battle keeps physical polling");
    inputs.resetHistory();
    require(inputs.apply(0, physical, false, true, true).a == 0, "new battle clears held history");
    inputs.clear(true);
    inputs.resetEpisode();
    require(inputs.apply(0, physical, true, true, false).b == 7, "episode reset clears pending release");
    require(isValidInput(pressed, 1) && !isValidInput(pressed, 0), "duration validation");
    pressed.a = 2;
    require(!isValidInput(pressed, 1), "commands contain button intent, not held counters");
    return 0;
}
