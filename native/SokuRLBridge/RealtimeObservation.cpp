#include "RealtimeObservation.hpp"
#include "LocalStart.hpp"
#include <BattleManager.hpp>
#include <Windows.h>
#include <cstring>
#include <cwchar>

namespace {
using namespace SokuRLBridge;
HANDLE g_handle = nullptr;
RealtimeHistory *g_history = nullptr;
// The game owns these pointers and is stopped at its own update boundary.
// Capture faults are reported in the slot; they never escape into the game.
bool copyMemory(std::uint32_t address, void *target, std::uint32_t size) {
    __try {
        std::memcpy(target, reinterpret_cast<const void *>(address), size);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}
}

namespace SokuRLBridge {
bool initializeOfflineObservation() {
    wchar_t value[16]{};
    const auto length = GetEnvironmentVariableW(L"SOKURL_OFFLINE_SNAPSHOT", value, _countof(value));
    if (!length) return GetLastError() == ERROR_ENVVAR_NOT_FOUND;
    // Observation-only opt-in. Never enable the realtime input controller or
    // accept this transport in an unpaused/local-human/network session.
    if (length != 1 || value[0] != L'1' || g_history ||
        environmentValue(L"SOKURL_VS_BOOTSTRAP", 0) != 1 ||
        environmentValue(L"SOKURL_VS_PAUSE_AT_START", 0) != 1 ||
        GetEnvironmentVariableW(L"SOKURL_NETWORK_ROLE", value, _countof(value))) return false;
    return initializeRealtimeObservation();
}

bool initializeRealtimeObservation() {
    wchar_t name[64]{};
    swprintf_s(name, L"Local\\SokuRLRealtimeHistory_%lu", GetCurrentProcessId());
    g_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, sizeof(RealtimeHistory), name);
    if (!g_handle) return false;
    g_history = static_cast<RealtimeHistory *>(MapViewOfFile(g_handle, FILE_MAP_ALL_ACCESS,
        0, 0, sizeof(RealtimeHistory)));
    if (!g_history) { closeRealtimeObservation(); return false; }
    // New mapping pages are zeroed by Windows. Do not touch unused snapshot
    // capacity: physical memory use follows the data actually captured.
    LARGE_INTEGER frequency{};
    QueryPerformanceFrequency(&frequency);
    g_history->magic = 0x48524B53;
    g_history->version = 1;
    g_history->size = sizeof(RealtimeHistory);
    g_history->capacity = REALTIME_HISTORY_CAPACITY;
    g_history->frequency = frequency.QuadPart;
    g_history->alive = 1;
    return true;
}

void closeRealtimeObservation() {
    if (g_history) {
        InterlockedExchange(reinterpret_cast<volatile LONG *>(&g_history->alive), 0);
        UnmapViewOfFile(g_history);
        g_history = nullptr;
    }
    if (g_handle) { CloseHandle(g_handle); g_handle = nullptr; }
}

void publishRealtimeObservation(const RawFrameState &raw) {
    if (!g_history) return;
    LARGE_INTEGER before{}, after{};
    QueryPerformanceCounter(&before);
    const auto next = static_cast<std::uint32_t>(g_history->published)+1U;
    auto &slot = g_history->frames[(next-1U)%REALTIME_HISTORY_CAPACITY];
    InterlockedIncrement(&slot.guard);
    MemoryBarrier();
    slot.match = raw.segmentId;
    slot.frame = raw.frameId;
    const auto &manager = SokuLib::getBattleMgr();
    slot.scores[0] = static_cast<unsigned char>(manager.leftCharacterManager.score);
    slot.scores[1] = static_cast<unsigned char>(manager.rightCharacterManager.score);
    slot.raw = raw;
    captureRenderState(slot.render);
    capturePrivilegedSnapshot(slot.memory, copyMemory);
    QueryPerformanceCounter(&after);
    slot.captureTicks = after.QuadPart-before.QuadPart;
    MemoryBarrier();
    InterlockedIncrement(&slot.guard);
    InterlockedExchange(&g_history->published, static_cast<LONG>(next));
}
}
