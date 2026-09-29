#include "NetworkState.hpp"
#include <Windows.h>
#include <cwchar>

namespace {
HANDLE g_handle = nullptr;
SokuRLBridge::NetworkState *g_state = nullptr;
constexpr unsigned HISTORY_CAPACITY = 256;
#pragma pack(push, 4)
struct NetworkHistory {
    unsigned magic, version, size, capacity;
    volatile LONG sequence;
    unsigned alive;
    std::uint64_t written;
    SokuRLBridge::NetworkState entries[HISTORY_CAPACITY];
};
#pragma pack(pop)
static_assert(offsetof(NetworkHistory, entries) == 32, "network history header");
HANDLE g_historyHandle = nullptr;
NetworkHistory *g_history = nullptr;
bool battle(unsigned scene) { return scene == 13 || scene == 14; }
void beginWrite() { InterlockedIncrement(&g_state->sequence); MemoryBarrier(); }
void endWrite() { MemoryBarrier(); InterlockedIncrement(&g_state->sequence); }
void appendHistory() {
    InterlockedIncrement(&g_history->sequence);
    MemoryBarrier();
    g_history->entries[g_history->written % HISTORY_CAPACITY] = *g_state;
    ++g_history->written;
    MemoryBarrier();
    InterlockedIncrement(&g_history->sequence);
}
}

namespace SokuRLBridge {
bool initializeNetworkState()
{
    wchar_t name[64]{};
    swprintf_s(name, L"Local\\SokuRLNetwork_%lu", GetCurrentProcessId());
    g_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, sizeof(NetworkState), name);
    if (!g_handle)
        return false;
    g_state = static_cast<NetworkState *>(MapViewOfFile(g_handle,
        FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NetworkState)));
    if (!g_state) {
        CloseHandle(g_handle);
        g_handle = nullptr;
        return false;
    }
    *g_state = {};
    g_state->magic = 0x54454E53;
    g_state->version = 2;
    g_state->size = sizeof(NetworkState);
    g_state->connected = 1;
    g_state->localSeat = UINT32_MAX;
    swprintf_s(name, L"Local\\SokuRLNetworkHistory_%lu", GetCurrentProcessId());
    g_historyHandle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, sizeof(NetworkHistory), name);
    if (g_historyHandle)
        g_history = static_cast<NetworkHistory *>(MapViewOfFile(g_historyHandle,
            FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NetworkHistory)));
    if (!g_history) {
        closeNetworkState();
        return false;
    }
    // A newly created page-file mapping is already zero-filled.
    g_history->magic = 0x484E4B53;
    g_history->version = 2;
    g_history->size = sizeof(NetworkHistory);
    g_history->capacity = HISTORY_CAPACITY;
    g_history->alive = 1;
    return true;
}

void closeNetworkState()
{
    if (g_history) {
        InterlockedIncrement(&g_history->sequence);
        MemoryBarrier();
        g_history->alive = 0;
        MemoryBarrier();
        InterlockedIncrement(&g_history->sequence);
        UnmapViewOfFile(g_history);
        g_history = nullptr;
    }
    if (g_historyHandle) {
        CloseHandle(g_historyHandle);
        g_historyHandle = nullptr;
    }
    if (g_state) {
        beginWrite();
        g_state->connected = 0;
        endWrite();
        UnmapViewOfFile(g_state);
        g_state = nullptr;
    }
    if (g_handle) {
        CloseHandle(g_handle);
        g_handle = nullptr;
    }
}

void observeNetworkScene(std::uint32_t scene)
{
    if (!g_state || g_state->scene == scene)
        return;
    beginWrite();
    if (battle(scene) && !battle(g_state->scene)) {
        ++g_state->match;
        g_state->updates = 0;
        g_state->scores[0] = g_state->scores[1] = 0;
    }
    g_state->scene = scene;
    g_state->localSeat = (scene == 8 || scene == 10 || scene == 13) ? 0 :
        (scene == 9 || scene == 11 || scene == 14) ? 1 : UINT32_MAX;
    endWrite();
    appendHistory();
}

std::uint64_t nextNetworkUpdate() { return g_state->updates + 1; }
std::uint32_t networkMatch() { return g_state->match; }
const NetworkState &currentNetworkState() { return *g_state; }

void publishNetworkState(const RawFrameState &raw, unsigned leftScore, unsigned rightScore)
{
    beginWrite();
    g_state->updates = raw.frameId;
    g_state->scores[0] = leftScore;
    g_state->scores[1] = rightScore;
    g_state->raw = raw;
    captureRenderState(g_state->render);
    endWrite();
    appendHistory();
}
}
