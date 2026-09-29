#include "NetworkInput.hpp"
#include "NetworkInputEvents.hpp"
#include "NetworkState.hpp"
#include <InputManager.hpp>
#include <Windows.h>
#include <cwchar>
#include <deque>

namespace {
using namespace SokuRLBridge;
HANDLE g_handle = nullptr;
NetworkInputBlock *g_block = nullptr;
struct Request {
    unsigned sequence, command, match, round, duration;
    std::uint64_t observed, target;
    LogicalInput input;
};
std::deque<Request> g_pending;
LogicalInput g_held{};
bool g_owned = false, g_hasObservation = false;
unsigned g_match = 0, g_round = 0;
unsigned g_phase = 0;
std::uint64_t g_lastObservation = 0, g_expires = 0;

unsigned load(const unsigned *p) {
    return InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(const_cast<unsigned *>(p)), 0, 0);
}
void record(const Request &request, unsigned kind, std::uint64_t at) {
    appendNetworkInputEvent({request.sequence, kind, request.command, request.match,
        request.round, request.duration, request.observed, request.target, at});
}
void clear(std::uint64_t at) {
    for (const auto &request : g_pending) record(request, 10, at);
    g_pending.clear();
    g_held = {};
    g_hasObservation = false;
    g_expires = 0;
}
bool valid(const LogicalInput &v) {
    return v.horizontalAxis >= -1 && v.horizontalAxis <= 1 &&
        v.verticalAxis >= -1 && v.verticalAxis <= 1 &&
        (v.a == 0 || v.a == 1) && (v.b == 0 || v.b == 1) &&
        (v.c == 0 || v.c == 1) && (v.d == 0 || v.d == 1) &&
        (v.changeCard == 0 || v.changeCard == 1) &&
        (v.spellcard == 0 || v.spellcard == 1);
}
void acknowledge(unsigned sequence, unsigned result) {
    g_block->result = result;
    MemoryBarrier();
    InterlockedExchange(reinterpret_cast<volatile LONG *>(&g_block->ackSequence), sequence);
}
}

namespace SokuRLBridge {
bool initializeNetworkInput() {
    wchar_t name[64]{};
    swprintf_s(name, L"Local\\SokuRLNetworkInput_%lu", GetCurrentProcessId());
    g_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, sizeof(NetworkInputBlock), name);
    if (!g_handle) return false;
    g_block = static_cast<NetworkInputBlock *>(MapViewOfFile(g_handle,
        FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NetworkInputBlock)));
    if (!g_block) { CloseHandle(g_handle); g_handle = nullptr; return false; }
    *g_block = {};
    g_block->magic = 0x494E4B53;
    g_block->version = 1;
    g_block->size = sizeof(NetworkInputBlock);
    g_block->injectedAt = NO_FRAME;
    if (!initializeNetworkInputEvents()) { closeNetworkInput(); return false; }
    return true;
}

void closeNetworkInput() {
    clear(NO_FRAME);
    closeNetworkInputEvents();
    if (g_block) { UnmapViewOfFile(g_block); g_block = nullptr; }
    if (g_handle) { CloseHandle(g_handle); g_handle = nullptr; }
    g_owned = false;
}

void serviceNetworkInput() {
    if (!g_block) return;
    const auto &state = currentNetworkState();
    const bool battle = (state.scene == 13 || state.scene == 14) && state.updates;
    const bool finished = battle && (state.scores[0] >= 2 || state.scores[1] >= 2);
    const bool fighting = battle && !finished &&
        state.raw.p1.hp > 0 && state.raw.p2.hp > 0;
    const unsigned phase = finished ? 3 : fighting ? 2 : battle ? 1 : 0;
    if ((!fighting && !finished) || phase != g_phase ||
        state.match != g_match || state.raw.roundId != g_round) {
        clear(state.updates);
        g_match = state.match;
        g_round = state.raw.roundId;
    }
    g_phase = phase;
    const auto sequence = load(&g_block->requestSequence);
    if (sequence == load(&g_block->ackSequence)) return;
    MemoryBarrier();
    const auto command = g_block->command;
    const Request request{sequence, command, g_block->match, g_block->round, g_block->duration,
        g_block->observed, g_block->target, g_block->input};
    if (command == 2) {
        clear(state.updates); g_owned = false;
        record(request, 2, state.updates); acknowledge(sequence, 2); return;
    }
    unsigned result = 1; // Accepted; injection is reported separately.
    const auto &keys = request.input;
    const bool confirm = command == 3 && request.duration == 1 && keys.a == 1 &&
        keys.horizontalAxis == 0 && keys.verticalAxis == 0 && keys.b == 0 &&
        keys.c == 0 && keys.d == 0 && keys.changeCard == 0 && keys.spellcard == 0;
    if ((command != 1 && !confirm) || !valid(request.input) || request.duration < 1 || request.duration > 120)
        result = 3;
    else if (!(confirm ? finished : fighting) || request.match != state.match || request.round != state.raw.roundId)
        result = 4;
    else if (request.observed > state.updates || request.observed > UINT64_MAX - 125 ||
        request.target != request.observed + 5)
        result = 3;
    else if (request.target < state.updates)
        result = 5; // Inference/transport missed the requested input boundary.
    else if (g_hasObservation && (request.observed < g_lastObservation ||
        request.observed - g_lastObservation < 3))
        result = 6;
    else if (g_pending.size() >= 64)
        result = 7;
    if (result == 1) {
        g_pending.push_back(request);
        g_lastObservation = request.observed;
        g_hasObservation = g_owned = true;
    }
    record(request, result, state.updates);
    acknowledge(sequence, result);
}

void applyNetworkInput(SokuLib::KeymapManager *keyboard) {
    serviceNetworkInput();
    const auto &state = currentNetworkState();
    if (!g_owned || (state.scene != 13 && state.scene != 14)) return;
    // Loading changes +0x208 from the menu keyboard to the local profile input
    // (0x43F045/0x43F08A). The sender reads this object's +0x62 packed keys at
    // 0x454CA9/0x454CCB; the peer receive objects at +0xF8/+0x174 are separate.
    const auto network = *reinterpret_cast<unsigned char **>(0x008986A0);
    if (!network)
        return;
    // Victory dialogue reads its local input cluster (0x89A248) at 0x4610CD,
    // not the battle sender. Its update completes at the call at 0x43E55F.
    // Only command 3 can be queued after a match result; no peer input is changed.
    const bool finished = state.scores[0] >= 2 || state.scores[1] >= 2;
    const auto target = finished ? reinterpret_cast<SokuLib::KeymapManager *>(0x0089A248) :
        *reinterpret_cast<SokuLib::KeymapManager **>(network + 0x208);
    if (keyboard != target)
        return;
    if (state.updates >= g_expires) g_held = {};
    while (!g_pending.empty() && g_pending.front().target <= state.updates) {
        const auto request = g_pending.front();
        g_pending.pop_front();
        // The network input hook can be skipped while waiting for the peer.
        // Never apply a command after its specified boundary.
        if (request.target != state.updates) { record(request, 9, state.updates); continue; }
        g_held = request.input;
        g_expires = request.target + request.duration;
        InterlockedIncrement(&g_block->statusSequence);
        MemoryBarrier();
        g_block->injectedSequence = request.sequence;
        g_block->injectedAt = state.updates;
        MemoryBarrier();
        InterlockedIncrement(&g_block->statusSequence);
        record(request, 8, state.updates);
    }
    keyboard->input = {g_held.horizontalAxis, g_held.verticalAxis, g_held.a,
        g_held.b, g_held.c, g_held.d, g_held.changeCard, g_held.spellcard};
}
}
