#include "../ControlBlock.hpp"
using namespace SokuRLBridge;
constexpr auto step = CommandType::StepWithControlledInputs;
static_assert(controlledStepMask(step, 1) == 1, "only player one is controlled");
static_assert(controlledStepMask(step, 2) == 2, "only player two is controlled");
static_assert(controlledStepMask(step, 3) == 3, "both players are controlled");
static_assert(controlledStepMask(step, 0) == 0, "empty control is invalid");
static_assert(controlledStepMask(step, 4) == 0, "nonexistent player is invalid");
static_assert(controlledStepMask(step, 0x100000001ULL) == 0, "mask cannot truncate");
static_assert(controlledStepMask(CommandType::StepWithInputs, 0) == 3, "legacy joint command");
static_assert(controlledStepMask(CommandType::Run, 3) == 0, "other commands do not take control");
int main() { return 0; }
