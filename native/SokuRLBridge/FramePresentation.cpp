#include "FramePresentation.hpp"

#include <BattleMode.hpp>
#include <SokuAddresses.hpp>
#include <Windows.h>
#include <cstdint>
#include <cstring>

namespace {
using WaitForSingleObjectFunction = DWORD (WINAPI *)(HANDLE, DWORD);
constexpr std::uint32_t LOCAL_BATTLE_SCENE = 5;
constexpr DWORD RENDER_BRANCH = 0x00407FAE;
constexpr DWORD SKIP_RENDER_PATH = 0x00408048;
constexpr DWORD FRAME_WAIT_CALL_OPERAND = 0x00419689;
constexpr DWORD WAIT_FOR_SINGLE_OBJECT_IAT = 0x008570A0;
bool g_headlessRender = false;
bool g_captureImages = false;
bool g_renderPending = true;
bool g_unlimitedPacing = false;

void __declspec(naked) renderBranchDispatch()
{
    __asm {
        // Preserve the original JNE first. The injected JMP does not alter EFLAGS.
        jne originalSkip
        cmp dword ptr ds:[008A0044h], 5
        je localBattle
        // Network AI engines can omit drawing without changing their update
        // clock, input transport, menus or the peer's visible game.
        cmp byte ptr [g_headlessRender], 0
        je originalRender
        cmp dword ptr ds:[008A0044h], 13
        je originalSkip
        cmp dword ptr ds:[008A0044h], 14
        je originalSkip
        jmp originalRender
    localBattle:
        cmp byte ptr [g_headlessRender], 0
        jne originalSkip
        cmp byte ptr [g_captureImages], 0
        je originalRender
        cmp byte ptr [g_renderPending], 0
        je originalSkip
    originalRender:
        push 00407FB4h
        ret
    originalSkip:
        push 00408048h
        ret
    }
}

bool installHeadlessRenderHook()
{
    if (!g_headlessRender && !g_captureImages)
        return true;

    auto *branch = reinterpret_cast<unsigned char *>(RENDER_BRANCH);
    if (branch[0] != 0x0F || branch[1] != 0x85)
        return false;
    std::int32_t originalDisplacement = 0;
    std::memcpy(&originalDisplacement, branch + 2, sizeof(originalDisplacement));
    if (RENDER_BRANCH + 6 + originalDisplacement != SKIP_RENDER_PATH)
        return false;

    const auto displacement = static_cast<std::int32_t>(
        reinterpret_cast<std::uintptr_t>(renderBranchDispatch) - (RENDER_BRANCH + 5));
    branch[0] = 0xE9;
    std::memcpy(branch + 1, &displacement, sizeof(displacement));
    branch[5] = 0x90;
    return true;
}

DWORD WINAPI framePacingWait(HANDLE object, DWORD timeout)
{
    if (g_unlimitedPacing &&
        *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID) == LOCAL_BATTLE_SCENE &&
        SokuLib::mainMode == SokuLib::BATTLE_MODE_VSPLAYER)
        timeout = 0;
    return WaitForSingleObject(object, timeout);
}

WaitForSingleObjectFunction g_framePacingWait = framePacingWait;

bool installUnlimitedPacingHook()
{
    if (!g_unlimitedPacing)
        return true;
    auto *operand = reinterpret_cast<DWORD *>(FRAME_WAIT_CALL_OPERAND);
    if (*operand != WAIT_FOR_SINGLE_OBJECT_IAT)
        return false;
    *operand = reinterpret_cast<DWORD>(&g_framePacingWait);
    return true;
}

}

namespace SokuRLBridge {
bool installFramePresentation(bool headless, bool captureImages, bool unlimitedPacing)
{
    g_headlessRender = headless;
    g_captureImages = captureImages;
    g_unlimitedPacing = unlimitedPacing;
    const bool renderInstalled = installHeadlessRenderHook();
    const bool pacingInstalled = installUnlimitedPacingHook();
    return renderInstalled && pacingInstalled;
}

void setRenderPending(bool pending)
{
    g_renderPending = pending;
}
}
