#include "ControlBlock.hpp"
#include "FrameState.hpp"
#include "FrameRecords.hpp"
#include "FramePresentation.hpp"
#include "ImageCapture.hpp"
#include "AudioMute.hpp"
#include "SceneReset.hpp"
#include "NetworkStart.hpp"
#include "NetworkState.hpp"
#include "NetworkInput.hpp"
#include "NetworkSelection.hpp"
#include "LocalStart.hpp"
#include "HeldInput.hpp"
#include "ControlledInput.hpp"

#include <BattleManager.hpp>
#include <BattleMode.hpp>
#include <Character.hpp>
#include <Hash.hpp>
#include <InputManager.hpp>
#include <PracticeSettings.hpp>
#include <Scenes.hpp>
#include <SokuAddresses.hpp>
#include <Tamper.hpp>
#include <UnionCast.hpp>
#include <VTables.hpp>
#include <Weather.hpp>

#include <Windows.h>
#include <algorithm>
#include <cwchar>
#include <cstring>
#include <optional>

namespace
{
using SokuRLBridge::stateHash;
using SokuRLBridge::isPracticeGameplay;
using SokuRLBridge::isLocalVersusGameplay;
using SokuRLBridge::isSupportedGameplay;
using SokuRLBridge::isValidInput;
using SokuRLBridge::environmentValue;
using SokuRLBridge::captureState;
using SokuRLBridge::load32;
using SokuRLBridge::store32;
using SokuRLBridge::beginStatusWrite;
using SokuRLBridge::endStatusWrite;
using SokuRLBridge::applySimpleState;
using SokuRLBridge::isValidSimplePlayerState;

using SetInputsMethod = void (SokuLib::KeymapManager::*)();
using BattleProcessMethod = int (SokuLib::Battle::*)();
using BattleRenderMethod = int (SokuLib::Battle::*)();
using BattleManagerProcessMethod = int (SokuLib::BattleManager::*)();
using SelectProcessMethod = int (SokuLib::Select::*)();
using TitleProcessMethod = int (SokuLib::Title::*)();

constexpr DWORD KEYMAP_SET_INPUTS_HOOK = 0x0040A45D;
constexpr DWORD INPUT_CLUSTER_UPDATE_HOOK = 0x0043E55F;
constexpr DWORD P1_KEYMAP_MANAGER_PTR = 0x008989A0;
constexpr DWORD P2_KEYMAP_MANAGER_PTR = 0x0089918C;
constexpr DWORD FALLBACK_KEY_MANAGER = 0x008986A8;

SokuRLBridge::ControlBlock *g_control = nullptr;
SetInputsMethod g_originalSetInputs = nullptr;
SetInputsMethod g_originalClusterInputs = nullptr;
BattleProcessMethod g_originalBattleProcess = nullptr;
BattleRenderMethod g_originalBattleRender = nullptr;
bool g_captureImages = false;
bool g_captureStateOnly = false;
BattleManagerProcessMethod g_originalBattleManagerProcess = nullptr;
SelectProcessMethod g_originalSelectProcess = nullptr;
TitleProcessMethod g_originalTitleProcess = nullptr;

std::uint32_t g_lastCommandSeq = 0;
std::uint32_t g_segmentId = 0;
std::uint64_t g_currentFrame = 0;
std::uint32_t g_stepsRemaining = 0;
bool g_paused = false;
bool g_inSimulationUpdate = false;
bool g_battleActive = false;
bool g_checkpointArmed = false;
bool g_checkpointSeedRequested = false;
std::uint32_t g_requestedCheckpointSeed = 0;

SokuRLBridge::ControlledInput g_inputs(SokuRLBridge::advanceHeldInput);
bool g_vsBootstrapArmed = false;
bool g_vsBootstrapComplete = false;
bool g_episodeResetRequested = false;
bool g_headlessRender = false;
bool g_unlimitedPacing = false;
SokuRLBridge::LocalStart g_localStart;

SokuRLBridge::CheckpointIdentity g_checkpoint{};
std::optional<SokuRLBridge::FrameRecords> g_records;

SokuRLBridge::LogicalInput toLogicalInput(const SokuLib::KeyInput &input)
{
    return {input.horizontalAxis, input.verticalAxis, input.a, input.b, input.c, input.d,
        input.changeCard, input.spellcard};
}

SokuLib::KeyInput toKeyInput(const SokuRLBridge::LogicalInput &input)
{
    return {input.horizontalAxis, input.verticalAxis, input.a, input.b, input.c, input.d,
        input.changeCard, input.spellcard};
}

void publishResult(SokuRLBridge::ResultCode result)
{
    store32(&g_control->resultCode, static_cast<std::uint32_t>(result));
}

void acknowledge(std::uint32_t sequence)
{
    MemoryBarrier();
    store32(&g_control->ackSeq, sequence);
}

void clearControlledInput(SokuRLBridge::ResultCode result, bool neutral)
{
    g_inputs.clear(neutral);
    store32(&g_control->inputFramesRemaining, 0);
    publishResult(result);
}

void consumeCommand(bool gameplay)
{
    const auto sequence = load32(&g_control->commandSeq);
    if (sequence == g_lastCommandSeq)
        return;
    MemoryBarrier();
    const auto type = static_cast<SokuRLBridge::CommandType>(load32(&g_control->commandType));
    const auto input = g_control->commandInput;
    const auto inputP2 = g_control->commandInputP2;
    const auto duration = g_control->durationFrames;
    const auto argument = g_control->commandArgument;
    const auto stepInputMask = SokuRLBridge::controlledStepMask(type, argument);
    g_lastCommandSeq = sequence;

    if (type == SokuRLBridge::CommandType::Release) {
        clearControlledInput(SokuRLBridge::ResultCode::Released, gameplay);
    } else if (!gameplay) {
        publishResult(SokuRLBridge::ResultCode::NotInGameplay);
    } else if (type == SokuRLBridge::CommandType::ResetEpisode &&
        g_vsBootstrapArmed && isLocalVersusGameplay() && argument < 0xFFFFFFFFULL) {
        g_localStart.seed = static_cast<std::uint32_t>(argument);
        g_localStart.seedRequested = true;
        g_episodeResetRequested = true;
        g_paused = true;
        g_stepsRemaining = 0;
        clearControlledInput(SokuRLBridge::ResultCode::Restarting, true);
    } else if (type == SokuRLBridge::CommandType::Input && isValidInput(input, duration)) {
        g_inputs.request(input, {}, 1, duration);
        store32(&g_control->inputFramesRemaining, duration);
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::Run) {
        g_paused = false;
        g_stepsRemaining = 0;
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::Pause) {
        g_paused = true;
        g_stepsRemaining = 0;
        publishResult(SokuRLBridge::ResultCode::Complete);
    } else if (type == SokuRLBridge::CommandType::StepFrames && duration > 0 && duration <= 10000) {
        g_paused = true;
        g_stepsRemaining = duration;
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (stepInputMask &&
        isValidInput(input, 1) && isValidInput(inputP2, 1)) {
        g_inputs.request(input, inputP2, stepInputMask, 1);
        g_paused = true;
        g_stepsRemaining = 1;
        store32(&g_control->inputFramesRemaining, 1);
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::ApplySimpleState && g_paused &&
        !g_stepsRemaining && isValidSimplePlayerState(g_control->commandPatch.p1) &&
        isValidSimplePlayerState(g_control->commandPatch.p2)) {
        auto &manager = SokuLib::getBattleMgr();
        applySimpleState(manager, g_control->commandPatch);
        const auto patched = captureState(&manager, g_currentFrame, g_segmentId, g_inputs.effective);
        g_records->publishLatest(patched, g_stepsRemaining);
        g_records->pushRing(patched);
        publishResult(SokuRLBridge::ResultCode::Complete);
    } else if (type == SokuRLBridge::CommandType::EstablishCheckpoint && !g_battleActive) {
        g_checkpointArmed = true;
        g_paused = true;
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::EstablishCheckpoint) {
        publishResult(SokuRLBridge::ResultCode::CheckpointRestoreUnsupported);
    } else if (type == SokuRLBridge::CommandType::GotoFrame) {
        (void)argument;
        publishResult(SokuRLBridge::ResultCode::CheckpointRestoreUnsupported);
    } else {
        publishResult(SokuRLBridge::ResultCode::InvalidCommand);
    }
    acknowledge(sequence);
}

int playerIndexFor(SokuLib::KeymapManager *self)
{
    if (!isSupportedGameplay())
        return -1;
    if (self == *reinterpret_cast<SokuLib::KeymapManager **>(P1_KEYMAP_MANAGER_PTR))
        return 0;
    if (self == *reinterpret_cast<SokuLib::KeymapManager **>(P2_KEYMAP_MANAGER_PTR))
        return 1;
    auto &manager = SokuLib::getBattleMgr();
    const auto left = manager.leftCharacterManager.keyManager;
    const auto right = manager.rightCharacterManager.keyManager;
    if (left && left->keymapManager == self)
        return 0;
    if (right && right->keymapManager == self)
        return 1;
    return -1;
}

void __fastcall inputClusterUpdate(SokuLib::KeymapManager *self)
{
    (self->*g_originalClusterInputs)();
    if (g_control)
        SokuRLBridge::applyNetworkInput(self);
}

void __fastcall keymapManagerSetInputs(SokuLib::KeymapManager *self)
{
    (self->*g_originalSetInputs)();
    if (!g_control)
        return;
    const auto scene = *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID);
    SokuRLBridge::observeNetworkScene(scene);
    SokuRLBridge::applyNetworkInput(self);
    const bool networkSelection = scene == SokuLib::SCENE_SELECTSV ||
        scene == SokuLib::SCENE_SELECTCL;
    // Network startup selects the local keyboard. Inject before the original
    // input routine packs its bits; never write the network peer's key manager.
    const bool localNetworkKeyboard = networkSelection &&
        self == reinterpret_cast<SokuLib::KeymapManager *>(FALLBACK_KEY_MANAGER);
    if (scene == SokuLib::SCENE_SELECT || localNetworkKeyboard) {
        const auto sequence = load32(&g_control->commandSeq);
        if (sequence != g_lastCommandSeq) {
            MemoryBarrier();
            const auto type = static_cast<SokuRLBridge::CommandType>(load32(&g_control->commandType));
            if (type == SokuRLBridge::CommandType::MenuConfirm) {
                self->input.a = 1;
                g_lastCommandSeq = sequence;
                publishResult(SokuRLBridge::ResultCode::Complete);
                acknowledge(sequence);
            } else if (networkSelection && type == SokuRLBridge::CommandType::MenuChooseCharacter) {
                const auto seat = SokuRLBridge::currentNetworkState().localSeat;
                const auto character = g_control->commandArgument;
                if (seat > 1 || character > 19 || !SokuLib::currentScene) {
                    publishResult(SokuRLBridge::ResultCode::InvalidCommand);
                } else {
                    const auto &select = SokuLib::currentScene->to<SokuLib::Select>();
                    const auto selected = seat ? SokuLib::gameParams.rightPlayerInfo.character :
                        SokuLib::gameParams.leftPlayerInfo.character;
                    const auto stage = seat ? select.rightSelectionStage : select.leftSelectionStage;
                    // Use the local menu's original packed inputs so the peer sees
                    // the same selection. Never overwrite either player's character.
                    if (selected == character)
                        self->input.a = 1;
                    else if (stage != 0)
                        self->input.b = 1;
                    else {
                        const auto cursor = seat ? select.rightCursor.cursorPos : select.leftCursor.cursorPos;
                        self->input.horizontalAxis = SokuRLBridge::selectionDirection(
                            cursor, SokuRLBridge::characterCursor(static_cast<unsigned>(character)));
                    }
                    publishResult(SokuRLBridge::ResultCode::Complete);
                }
                g_lastCommandSeq = sequence;
                acknowledge(sequence);
            } else if (!networkSelection && type == SokuRLBridge::CommandType::EstablishCheckpoint) {
                const auto seed = g_control->commandArgument;
                if (SokuLib::practiceSettings)
                    SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
                g_checkpointSeedRequested = seed != SokuRLBridge::NO_FRAME;
                if (g_checkpointSeedRequested) {
                    g_requestedCheckpointSeed = static_cast<std::uint32_t>(seed);
                    SokuLib::gameParams.randomSeed = g_requestedCheckpointSeed;
                }
                g_checkpointArmed = true;
                g_paused = true;
                g_lastCommandSeq = sequence;
                publishResult(SokuRLBridge::ResultCode::Accepted);
                acknowledge(sequence);
            }
        }
        return;
    }
    // Offline joint-input requests must never reach either network player.
    // The manager hook rejects them, but keymap hooks can run before that hook.
    if (scene == SokuLib::SCENE_BATTLESV || scene == SokuLib::SCENE_BATTLECL)
        return;
    const auto player = playerIndexFor(self);
    if (player < 0)
        return;

    self->input = toKeyInput(g_inputs.apply(player, toLogicalInput(self->input),
        g_inSimulationUpdate, g_battleActive, g_paused));
}

int callSimulationUpdate(SokuLib::BattleManager *manager)
{
    if (isPracticeGameplay() && SokuLib::practiceSettings)
        SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
    g_inputs.clearEffective();
    g_inSimulationUpdate = true;
    const auto result = (manager->*g_originalBattleManagerProcess)();
    g_inSimulationUpdate = false;
    g_inputs.effective[0] = toLogicalInput(manager->leftCharacterManager.keyMap);
    g_inputs.effective[1] = toLogicalInput(manager->rightCharacterManager.keyMap);
    g_inputs.finishFrame(*g_control);
    return result;
}

int __fastcall battleManagerOnProcess(SokuLib::BattleManager *manager)
{
    if (!g_control)
        return (manager->*g_originalBattleManagerProcess)();
    const auto scene = static_cast<unsigned>(SokuLib::sceneId);
    SokuRLBridge::observeNetworkScene(scene);
    if (scene == SokuLib::SCENE_BATTLESV || scene == SokuLib::SCENE_BATTLECL) {
        // Offline pause, reset and joint-input commands cannot control netplay.
        store32(&g_control->inGameplay, 0);
        consumeCommand(false);
        const auto result = (manager->*g_originalBattleManagerProcess)();
        auto state = captureState(manager, SokuRLBridge::nextNetworkUpdate(), g_segmentId, g_inputs.effective);
        state.segmentId = SokuRLBridge::networkMatch();
        state.p1.input = toLogicalInput(manager->leftCharacterManager.keyMap);
        state.p2.input = toLogicalInput(manager->rightCharacterManager.keyMap);
        state.stateHash = stateHash(state);
        SokuRLBridge::publishNetworkState(state,
            static_cast<unsigned char>(manager->leftCharacterManager.score),
            static_cast<unsigned char>(manager->rightCharacterManager.score));
        SokuRLBridge::serviceNetworkInput();
        return result;
    }
    const bool gameplay = isSupportedGameplay();
    store32(&g_control->inGameplay, gameplay ? 1U : 0U);
    consumeCommand(gameplay);
    if (g_episodeResetRequested)
        return 0;
    if (!gameplay) {
        g_battleActive = false;
        g_records->invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
        return (manager->*g_originalBattleManagerProcess)();
    }
    if (!g_battleActive) {
        g_battleActive = true;
        g_currentFrame = 0;
        g_stepsRemaining = 0;
        g_inputs.resetHistory();
        if (g_checkpointArmed && isPracticeGameplay() && SokuLib::practiceSettings)
            SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
        if (g_checkpointArmed && g_checkpointSeedRequested)
            SokuLib::gameParams.randomSeed = g_requestedCheckpointSeed;
        const auto initial = captureState(manager, 0, g_segmentId, g_inputs.effective);
        if (g_checkpointArmed) {
            g_checkpointArmed = false;
            g_checkpointSeedRequested = false;
            g_checkpoint = SokuRLBridge::readCheckpointIdentity();
            g_records->recordInitial(initial);
            g_paused = true;
            store32(&g_control->validationState,
                static_cast<std::uint32_t>(SokuRLBridge::ValidationState::Unknown));
            g_control->lastVerifiedFrame = SokuRLBridge::NO_FRAME;
            g_control->firstDivergentFrame = SokuRLBridge::NO_FRAME;
            g_records->publishLatest(initial, g_stepsRemaining);
            g_records->pushRing(initial);
            publishResult(SokuRLBridge::ResultCode::Complete);
            store32(&g_control->runState,
                static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
            return 0;
        }
        g_records->publishLatest(initial, g_stepsRemaining);
        g_records->pushRing(initial);
    }
    if (load32(&g_control->checkpointValid) && !SokuRLBridge::identityMatchesCheckpoint(g_checkpoint))
        g_records->invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
    if (g_paused && !g_stepsRemaining) {
        store32(&g_control->runState, static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
        return 0;
    }

    const auto updates = g_stepsRemaining ? g_stepsRemaining : 1U;
    store32(&g_control->runState, static_cast<std::uint32_t>(
        g_stepsRemaining ? SokuRLBridge::RunState::Stepping : SokuRLBridge::RunState::Running));
    int result = 0;
    for (std::uint32_t i = 0; i < updates; ++i) {
        g_records->trim(g_currentFrame);
        result = callSimulationUpdate(manager);
        ++g_currentFrame;
        g_records->appendRecordedFrame(
            captureState(manager, g_currentFrame, g_segmentId, g_inputs.effective), g_stepsRemaining);
        SokuRLBridge::setRenderPending(true);
        if (g_stepsRemaining)
            --g_stepsRemaining;
        if (result > 0 && result < 4)
            break;
    }
    if (g_paused || g_stepsRemaining == 0 && updates > 1) {
        g_paused = true;
        g_stepsRemaining = 0;
        store32(&g_control->runState, static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
        publishResult(SokuRLBridge::ResultCode::Complete);
    }
    return result;
}

int __fastcall selectOnProcess(SokuLib::Select *select)
{
    if (g_checkpointArmed && SokuLib::practiceSettings)
        SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
    const auto result = (select->*g_originalSelectProcess)();
    if (g_checkpointArmed && g_checkpointSeedRequested)
        SokuLib::gameParams.randomSeed = g_requestedCheckpointSeed;
    return result;
}

int __fastcall titleOnProcess(SokuLib::Title *title)
{
    // Scene deletion is asynchronous. A new battle must not reuse global
    // resources while the previous battle is still destroying them.
    if (g_vsBootstrapArmed && !g_vsBootstrapComplete &&
        !SokuRLBridge::retiredBattleSceneDestroyed())
        return SokuLib::SCENE_TITLE;
    const auto result = (title->*g_originalTitleProcess)();
    SokuRLBridge::processNetworkStart(*title, result);
    if (!g_vsBootstrapArmed || g_vsBootstrapComplete)
        return result;

    if (!SokuRLBridge::configureLocalStart(g_localStart)) {
        publishResult(SokuRLBridge::ResultCode::TargetUnavailable);
        g_vsBootstrapArmed = false;
        return SokuLib::SCENE_TITLE;
    }
    if (g_localStart.pauseAtStart) {
        g_checkpointArmed = true;
        g_checkpointSeedRequested = g_localStart.seedRequested;
        g_requestedCheckpointSeed = g_localStart.seed;
        g_paused = true;
    }
    g_vsBootstrapComplete = true;
    return SokuLib::SCENE_LOADING;
}

int __fastcall battleOnProcess(SokuLib::Battle *battle)
{
    const auto result = (battle->*g_originalBattleProcess)();
    if (g_episodeResetRequested) {
        SokuRLBridge::retireBattleScene(battle);
        // Return through the engine's normal scene lifecycle. It destroys the
        // old battle asynchronously; titleOnProcess waits for that destruction
        // before loading the next one. The process and DLL stay alive.
        g_episodeResetRequested = false;
        g_vsBootstrapComplete = false;
        g_battleActive = false;
        g_checkpointArmed = false;
        g_checkpointSeedRequested = false;
        g_currentFrame = 0;
        ++g_segmentId;
        g_records->clear();
        g_inputs.resetEpisode();
        SokuRLBridge::setRenderPending(true);
        beginStatusWrite(g_control);
        g_control->inGameplay = 0;
        g_control->checkpointValid = 0;
        g_control->reconstructing = 0;
        g_control->currentFrame = 0;
        g_control->recordedFrames = 0;
        g_control->stepsRemaining = 0;
        g_control->ringReadSeq = 0;
        g_control->ringWriteSeq = 0;
        g_control->droppedFrames = 0;
        g_control->lastVerifiedFrame = SokuRLBridge::NO_FRAME;
        g_control->firstDivergentFrame = SokuRLBridge::NO_FRAME;
        endStatusWrite(g_control);
        SokuRLBridge::resetImageCapture();
        return SokuLib::SCENE_TITLE;
    }
    if (g_captureStateOnly && g_battleActive && g_control && isSupportedGameplay())
        SokuRLBridge::captureImage(g_currentFrame);
    return result;
}

int __fastcall battleOnRender(SokuLib::Battle *battle)
{
    const auto result = (battle->*g_originalBattleRender)();
    if (g_battleActive && g_control && isSupportedGameplay()) {
        SokuRLBridge::captureImage(g_currentFrame);
        SokuRLBridge::setRenderPending(false);
    }
    return result;
}

bool createMapping()
{
    auto *mapping = SokuRLBridge::openFrameMapping();
    if (!mapping) return false;
    g_control = &mapping->control;
    g_records.emplace(*mapping);
    return true;
}

void closeMapping()
{
    SokuRLBridge::closeNetworkState();
    SokuRLBridge::closeNetworkInput();
    SokuRLBridge::closeImageCapture();
    g_records.reset();
    SokuRLBridge::closeFrameMapping();
    g_control = nullptr;
}

bool installHooks()
{
    DWORD textProtection = 0;
    if (!VirtualProtect(reinterpret_cast<void *>(TEXT_SECTION_OFFSET), TEXT_SECTION_SIZE,
            PAGE_EXECUTE_WRITECOPY, &textProtection))
        return false;
    g_originalSetInputs = SokuLib::union_cast<SetInputsMethod>(
        SokuLib::TamperNearJmpOpr(KEYMAP_SET_INPUTS_HOOK, keymapManagerSetInputs));
    g_originalClusterInputs = SokuLib::union_cast<SetInputsMethod>(
        SokuLib::TamperNearJmpOpr(INPUT_CLUSTER_UPDATE_HOOK, inputClusterUpdate));
    const bool presentationInstalled = SokuRLBridge::installFramePresentation(
        g_headlessRender, g_captureImages, g_unlimitedPacing);
    DWORD ignored = 0;
    VirtualProtect(reinterpret_cast<void *>(TEXT_SECTION_OFFSET), TEXT_SECTION_SIZE,
        textProtection, &ignored);

    DWORD rdataProtection = 0;
    if (!VirtualProtect(reinterpret_cast<void *>(RDATA_SECTION_OFFSET), RDATA_SECTION_SIZE,
            PAGE_EXECUTE_WRITECOPY, &rdataProtection))
        return false;
    g_originalBattleManagerProcess = SokuLib::TamperDword(
        &SokuLib::VTable_BattleManager.onProcess, battleManagerOnProcess);
    g_originalSelectProcess = SokuLib::TamperDword(
        &SokuLib::VTable_Select.onProcess, selectOnProcess);
    g_originalTitleProcess = SokuLib::TamperDword(
        &SokuLib::VTable_Title.onProcess, titleOnProcess);
    const bool resetBarrierInstalled = SokuRLBridge::installSceneResetBarrier();
    if (g_captureImages)
        g_originalBattleRender = SokuLib::TamperDword(
            &SokuLib::VTable_Battle.onRender, battleOnRender);
    g_originalBattleProcess = SokuLib::TamperDword(
        &SokuLib::VTable_Battle.onProcess, battleOnProcess);
    VirtualProtect(reinterpret_cast<void *>(RDATA_SECTION_OFFSET), RDATA_SECTION_SIZE,
        rdataProtection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    return g_originalSetInputs && g_originalClusterInputs && presentationInstalled &&
        g_originalBattleManagerProcess && g_originalSelectProcess && g_originalTitleProcess &&
        (!g_captureImages || g_originalBattleRender) &&
        g_originalBattleProcess && resetBarrierInstalled;
}
}

extern "C" __declspec(dllexport) bool CheckVersion(const BYTE hash[16])
{
    return std::memcmp(hash, SokuLib::targetHash, sizeof(SokuLib::targetHash)) == 0;
}

extern "C" __declspec(dllexport) bool Initialize(HMODULE, HMODULE)
{
    static_assert(sizeof(void *) == 4, "SokuRLBridge must be built for Win32/x86");
    if (!createMapping())
        return false;
    if (!SokuRLBridge::initializeNetworkState() || !SokuRLBridge::initializeNetworkInput()) {
        closeMapping();
        return false;
    }
    g_headlessRender = environmentValue(L"SOKURL_HEADLESS_RENDER", 0) == 1;
    if (environmentValue(L"SOKURL_MUTE_AUDIO", g_headlessRender ? 1U : 0U) == 1 &&
        !SokuRLBridge::installAudioMute()) {
        closeMapping();
        return false;
    }
    const auto captureMode = environmentValue(L"SOKURL_CAPTURE_IMAGES", 0);
    g_captureImages = captureMode != 0;
    g_captureStateOnly = captureMode == 2;
    if (g_captureImages) {
        // Images require the original renderer, even when the window is unattended.
        g_headlessRender = g_captureStateOnly;
        if (!SokuRLBridge::initializeImageCapture(captureMode == 1)) {
            closeMapping();
            return false;
        }
    }
    g_unlimitedPacing = environmentValue(L"SOKURL_UNLIMITED_PACING", 0) == 1;
    g_vsBootstrapArmed = environmentValue(L"SOKURL_VS_BOOTSTRAP", 0) == 1;
    if (!SokuRLBridge::initializeNetworkStart(g_vsBootstrapArmed, g_unlimitedPacing)) {
        closeMapping();
        return false;
    }
    if (g_vsBootstrapArmed) {
        g_localStart = SokuRLBridge::readLocalStart();
    }
    const auto *commandLine = GetCommandLineW();
    if (commandLine && wcsstr(commandLine, L".rep")) {
        g_checkpointArmed = true;
        g_paused = true;
    }
    if (!installHooks()) {
        closeMapping();
        return false;
    }
    return true;
}

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_DETACH)
        closeMapping();
    return TRUE;
}
