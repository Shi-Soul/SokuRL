#include "RealtimeInput.hpp"
#include "ControlledInput.hpp"
#include "LocalStart.hpp"
#include <Windows.h>
#include <cwchar>
#include <optional>

namespace {
using namespace SokuRLBridge;
HANDLE g_handle = nullptr;
RealtimeInputBlock *g_block = nullptr;
std::optional<RealtimeInputSchedule> g_schedule;
unsigned g_seat = 0, g_acknowledged = 0;
RealtimeResult g_result = RealtimeResult::Idle;
long load(volatile long *value) { return InterlockedCompareExchange(value, 0, 0); }
}

namespace SokuRLBridge {
bool initializeRealtimeInput() {
    wchar_t value[16]{};
    const auto length = GetEnvironmentVariableW(L"SOKURL_REALTIME_SEAT", value, _countof(value));
    if (!length) return GetLastError() == ERROR_ENVVAR_NOT_FOUND;
    // Explicit opt-in; reject malformed configuration instead of choosing a seat.
    if (length != 1 || (value[0] != L'0' && value[0] != L'1') ||
        environmentValue(L"SOKURL_VS_PAUSE_AT_START", 0) ||
        environmentValue(L"SOKURL_UNLIMITED_PACING", 0) ||
        !environmentValue(L"SOKURL_VS_BOOTSTRAP", 0)) return false;
    g_seat = value[0] - L'0';
    wchar_t name[64]{};
    swprintf_s(name, L"Local\\SokuRLRealtimeInput_%lu", GetCurrentProcessId());
    g_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, sizeof(RealtimeInputBlock), name);
    if (!g_handle) return false;
    g_block = static_cast<RealtimeInputBlock *>(MapViewOfFile(g_handle, FILE_MAP_ALL_ACCESS,
        0, 0, sizeof(RealtimeInputBlock)));
    if (!g_block) { closeRealtimeInput(); return false; }
    *g_block = {};
    g_block->magic = 0x54524B53;
    g_block->version = 1;
    g_block->size = sizeof(RealtimeInputBlock);
    g_block->seat = g_seat;
    g_block->appliedAt = NO_FRAME;
    g_acknowledged = 0;
    g_result = RealtimeResult::Idle;
    g_schedule.emplace(g_seat);
    return true;
}

bool realtimeInputEnabled() { return g_block != nullptr; }

void closeRealtimeInput() {
    g_schedule.reset();
    if (g_block) { UnmapViewOfFile(g_block); g_block = nullptr; }
    if (g_handle) { CloseHandle(g_handle); g_handle = nullptr; }
}

void prepareRealtimeInput(ControlledInput &inputs, std::uint32_t match,
    std::uint32_t round, std::uint64_t frame) {
    if (!g_block) return;
    auto &schedule = *g_schedule;
    schedule.advance(match, round, frame);
    // Exactly one read attempt. A stalled/crashed writer can never stall the game.
    const auto before = load(&g_block->requestGuard);
    if (!(before & 1)) {
        MemoryBarrier();
        const auto request = g_block->command;
        MemoryBarrier();
        if (before == load(&g_block->requestGuard) && request.sequence && request.sequence != g_acknowledged) {
            g_result = schedule.submit(request);
            g_acknowledged = request.sequence;
        }
    }
    schedule.advance(match, round, frame);
    const auto held = schedule.input(g_seat, {});
    inputs.request(g_seat == 0 ? held : LogicalInput{},
        g_seat == 1 ? held : LogicalInput{}, 1U << g_seat, 1);
    InterlockedIncrement(&g_block->statusGuard);
    MemoryBarrier();
    g_block->acknowledged = g_acknowledged;
    g_block->result = static_cast<unsigned>(g_result);
    g_block->match = match;
    g_block->round = round;
    g_block->frame = frame;
    g_block->appliedSequence = schedule.appliedSequence();
    g_block->appliedAt = schedule.appliedAt();
    g_block->pending = schedule.size();
    g_block->held = held;
    MemoryBarrier();
    InterlockedIncrement(&g_block->statusGuard);
}
}
