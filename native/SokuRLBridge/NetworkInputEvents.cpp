#include "NetworkInputEvents.hpp"
#include <Windows.h>
#include <cstddef>
#include <cwchar>

namespace {
constexpr unsigned CAPACITY = 256;
#pragma pack(push, 4)
struct History {
    unsigned magic, version, size, capacity;
    volatile LONG sequence;
    unsigned alive;
    std::uint64_t written;
    SokuRLBridge::NetworkInputEvent entries[CAPACITY];
};
#pragma pack(pop)
static_assert(offsetof(History, entries) == 32, "input event history header");
HANDLE g_handle = nullptr;
History *g_history = nullptr;
}

namespace SokuRLBridge {
bool initializeNetworkInputEvents() {
    wchar_t name[64]{};
    swprintf_s(name, L"Local\\SokuRLNetworkInputEvents_%lu", GetCurrentProcessId());
    g_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(History), name);
    if (g_handle)
        g_history = static_cast<History *>(MapViewOfFile(g_handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(History)));
    if (!g_history) { closeNetworkInputEvents(); return false; }
    // New page-file mappings are zero-filled.
    g_history->magic = 0x454E4B53;
    g_history->version = 1;
    g_history->size = sizeof(History);
    g_history->capacity = CAPACITY;
    g_history->alive = 1;
    return true;
}

void appendNetworkInputEvent(const NetworkInputEvent &event) {
    if (!g_history) return;
    InterlockedIncrement(&g_history->sequence);
    MemoryBarrier();
    g_history->entries[g_history->written % CAPACITY] = event;
    ++g_history->written;
    MemoryBarrier();
    InterlockedIncrement(&g_history->sequence);
}

void closeNetworkInputEvents() {
    if (g_history) {
        InterlockedIncrement(&g_history->sequence);
        MemoryBarrier();
        g_history->alive = 0;
        MemoryBarrier();
        InterlockedIncrement(&g_history->sequence);
        UnmapViewOfFile(g_history);
        g_history = nullptr;
    }
    if (g_handle) { CloseHandle(g_handle); g_handle = nullptr; }
}
}
