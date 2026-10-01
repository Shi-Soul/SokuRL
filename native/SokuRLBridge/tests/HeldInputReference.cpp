#include "../HeldInput.hpp"

// Loaded as bytes by the x86 emulator; never injected into a game process.
extern "C" __declspec(dllexport) void decodeInput(
    const SokuRLBridge::LogicalInput *previous,
    const SokuRLBridge::LogicalInput *intent,
    SokuRLBridge::LogicalInput *output)
{
    *output = SokuRLBridge::advanceHeldInput(*previous, *intent);
}
