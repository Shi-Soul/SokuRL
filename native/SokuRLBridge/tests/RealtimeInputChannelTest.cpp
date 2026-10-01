#include "../RealtimeInput.hpp"
#include "../ControlledInput.hpp"
#include <Windows.h>
#include <cwchar>
#include <iostream>
#include <stdexcept>

using namespace SokuRLBridge;
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
LogicalInput decode(const LogicalInput &previous, const LogicalInput &intent) {
    auto result = intent;
    result.a = intent.a ? previous.a + 1 : 0;
    return result;
}

int main() {
    try {
        SetEnvironmentVariableW(L"SOKURL_REALTIME_SEAT", nullptr);
        require(initializeRealtimeInput() && !realtimeInputEnabled(), "training remains opt-out");
        SetEnvironmentVariableW(L"SOKURL_REALTIME_SEAT", L"0");
        SetEnvironmentVariableW(L"SOKURL_VS_BOOTSTRAP", L"1");
        SetEnvironmentVariableW(L"SOKURL_VS_PAUSE_AT_START", L"1");
        require(!initializeRealtimeInput(), "realtime cannot start paused");
        SetEnvironmentVariableW(L"SOKURL_VS_PAUSE_AT_START", L"0");
        SetEnvironmentVariableW(L"SOKURL_UNLIMITED_PACING", L"1");
        require(!initializeRealtimeInput(), "realtime cannot accelerate original pacing");
        SetEnvironmentVariableW(L"SOKURL_UNLIMITED_PACING", L"0");
        for (unsigned seat = 0; seat < 2; ++seat) {
            SetEnvironmentVariableW(L"SOKURL_REALTIME_SEAT", seat ? L"1" : L"0");
            require(initializeRealtimeInput(), "create channel");
            wchar_t name[64]{};
            swprintf_s(name, L"Local\\SokuRLRealtimeInput_%lu", GetCurrentProcessId());
            HANDLE handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
            require(handle != nullptr, "open producer mapping");
            auto *block = static_cast<RealtimeInputBlock *>(MapViewOfFile(handle, FILE_MAP_ALL_ACCESS,
                0, 0, sizeof(RealtimeInputBlock)));
            require(block && block->size == 168 && block->seat == seat, "protocol header");
            ControlledInput inputs(decode);
            const LogicalInput human{-7, 0, 42, 0, 0, 0, 0, 0};
            InterlockedIncrement(&block->requestGuard);
            block->command = {1, 7, 2, seat, 0, 0, 3, {0, 0, 1, 0, 0, 0, 0, 0}};
            InterlockedIncrement(&block->requestGuard);
            prepareRealtimeInput(inputs, 7, 2, 0);
            require(block->acknowledged == 1 && block->appliedSequence == 1 && block->appliedAt == 0,
                "producer request consumed and applied");
            // Emulate a writer which dies during its next update. The consumer
            // must continue indefinitely without waiting for an even guard.
            InterlockedIncrement(&block->requestGuard);
            block->command.sequence = 2;
            for (unsigned frame = 1; frame < 100000; ++frame) {
                prepareRealtimeInput(inputs, 7, 2, frame);
                require(inputs.apply(1-seat, human, true, true, false).a == 42, "human untouched");
                const auto ai = inputs.apply(seat, human, true, true, false);
                require((ai.a != 0) == (frame < 3), "crashed producer expires held input");
                require(block->acknowledged == 1 && block->frame == frame, "torn command not consumed");
            }
            InterlockedIncrement(&block->requestGuard);
            prepareRealtimeInput(inputs, 7, 3, 100000);
            require(block->result == static_cast<unsigned>(RealtimeResult::WrongContext), "old round rejected");
            require(block->held.a == 0 && block->pending == 0, "no old round actions");
            UnmapViewOfFile(block);
            CloseHandle(handle);
            closeRealtimeInput();
        }
    } catch (const std::exception &error) {
        closeRealtimeInput();
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
