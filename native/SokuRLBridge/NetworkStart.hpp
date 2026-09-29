#pragma once

namespace SokuLib { struct Title; }

namespace SokuRLBridge
{
// Optional startup through the game's own connection menu. No simulation control.
bool initializeNetworkStart(bool offlineBootstrap, bool unlimitedPacing);
void processNetworkStart(const SokuLib::Title &title, int nextScene);
}
