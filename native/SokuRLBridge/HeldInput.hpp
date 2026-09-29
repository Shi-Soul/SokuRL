#pragma once
#include "ControlBlock.hpp"

namespace SokuRLBridge {
// Apply one logical key state through th123's original replay input decoder.
LogicalInput advanceHeldInput(const LogicalInput &previous, const LogicalInput &intent);
}
