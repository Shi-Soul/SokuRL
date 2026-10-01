#pragma once

namespace SokuRLBridge {
// Install while the caller has made the game text section writable.
bool installFramePresentation(bool headless, bool captureImages, bool unlimitedPacing);
void setRenderPending(bool pending);
}
