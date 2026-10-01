# SokuRLBridge

Each th123 process publishes a PID-qualified shared-memory mapping:

```text
Local\SokuRLBridge_<pid>
```

This keeps commands and state isolated when MemoryPatch allows multiple game
instances. `MenuConfirm` injects one frame of the game's logical A/confirm
input while the local character-select scene is active; it does not synthesize
an OS key event or write a scene ID.

`SokuRLBridge.dll` is a Win32/x86 SWRSToys module for local Practice/VS Player logical
input injection, raw frame capture, simulation pause/step, and episode reset.
It does not synthesize Windows keyboard events or copy a battle-manager memory
snapshot. The native `GotoFrame` command is disabled.

## Simulation boundary

The canonical frame boundary is `SokuLib::VTable_BattleManager.onProcess`, the
same battle-manager process function used by ReplayInputView+ for pause and
frame step. A SokuRL frame increments only after one call to the original
battle process. Rendering, Python polling, and paused process calls do not
increment it.

The existing `KeymapManager::SetInputs` hook at `0x40A45D` remains the logical
input boundary. The hook records the effective P1 and P2 `KeyInput` values and
can replace both values with one atomic `StepWithInputs` command while paused.
Normal bridge actions still control only P1.

## Checkpoint model

`EstablishCheckpoint` is armed in the normal character-select scene. The bridge
fixes the requested match seed after `Select::onProcess`, captures frame zero
before the first battle-manager simulation update, switches the documented
Practice dummy setting to `DUMMY_STATE_2P_CONTROL`, and freezes there. This
makes both logical input streams controllable without OS input injection.
`ResetEpisode` returns through the title and loading scene lifecycle. It waits
for the old battle scene to finish destruction before creating the new battle.
The process stays alive; arbitrary state restoration is not supported.

The opt-in VS launcher uses a separate Title-context bootstrap. After the
original `Title::onProcess` runs, it selects fallback local input ownership,
calls `setBattleMode(BATTLE_MODE_VSPLAYER, BATTLE_SUBMODE_PLAYING2)`, initializes
both profiles and effective decks, and returns `SCENE_LOADING`. It is armed only
by the `SOKURL_VS_BOOTSTRAP` process environment variable from `tools/sokurl.py launch.command=vs`.

Action replay starts at frame zero and rebuilds action machines and object
lists through simulation. The diagnostic `ApplySimpleState` command can write
timer, weather, player position, speed, facing, HP, spirit, and card counters.
It cannot restore actions, animation state, hitstop, flags, hands, or objects.
The RL reset and action replay do not use this command to restore snapshots.
Historical reconstruction scripts and results are separate from the current
episode-reset contract. The bridge keeps the input history in shared memory.
It tracks the number of recorded frames without a second full-frame array.

## Shared memory ABI

ABI version 9 uses 4-byte packing. `StepWithControlledInputs` (command 13)
advances one simulation frame. Its argument selects player one (1), player two
(2), or both players (3). The original game processes inputs for unselected
players. Network games reject this offline command. Clients must reject older
versions, which do not support this selection. Version 8 combined `ResetEpisode`
with signed resource fields. The two development branches used version 7 for
different semantics.
Spirit fields are signed 32-bit ABI values
that preserve the game's signed 16-bit resource semantics, including transient
negative values. The structure sizes are unchanged from version 6:

- 10884-byte `ControlBlock` with sequenced commands and a seqlock-protected live
  `RawFrameState`.
- 10596-byte fixed-dimensional frame records with 64 object slots per player.
- A 512-record single-producer/single-consumer ring.
- A native 4096-frame short-range checkpoint history.

The producer never performs file I/O. `tools/debug_panel.py` drains the ring
and writes `data/raw/<session_id>/metadata.json`, `frames_000.csv`,
`inputs_000.csv`, and `manifest.json`. A lossless recording requires
`dropped_frames == 0`.

## Safety and invalidation

Battle commands are accepted only in local Practice, local VS Player, or replay battle.
Checkpoint arming uses the legitimate local character-select path for Practice
and the ReplayDnD command-line path for replay. Leaving battle or changing
selected characters, stage, start seed, or tracked Practice settings invalidates
the checkpoint. History capacity exhaustion also invalidates it instead of
silently wrapping.

## Replay mode

ReplayDnD owns command-line `.rep` loading. For a replay launch the bridge
automatically freezes the first replay BattleManager state at frame zero. Replay
mode uses the same state capture, dual-input stepping, simple-state patch, and
hash comparison path as Practice; it does not parse or replace ReplayDnD.

```powershell
.\.venv\python.exe tools\sokurl.py replay path\match.rep
.\.venv\python.exe tools\sokurl.py replay path\match.rep --frame 5000
```

The first form starts normal playback. The second fast-simulates from replay
frame zero to the requested frame and leaves the process frozen there.
