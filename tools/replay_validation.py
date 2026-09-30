from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

import sokurl
from game_runtime.replay import launch_replay
from bridge_shared import BridgeClient, BridgeUnavailable, RawFrameState
from frame_validation import (
    InputPair,
    PracticeInstance,
    complex_state_diff,
    copy_state,
    input_tuple,
    state_diff,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"
SCENE_BATTLE = 5
BATTLE_MODE_VSPLAYER = 3
BATTLE_SUBMODE_REPLAY = 2


def launch_replay_checkpoint(replay: Path, timeout: float, unlimited: bool) -> PracticeInstance:
    return launch_replay(replay, timeout, unlimited, "diagnostic_state")


def play_full_replay(replay: Path, launch_timeout: float, timeout: float) -> tuple[list[RawFrameState], dict[str, object]]:
    if timeout <= 0:
        raise ValueError("replay playback timeout must be positive")
    instance = launch_replay_checkpoint(replay, launch_timeout, False)
    states = [instance.state]
    pid = instance.pid
    started = time.monotonic()
    sequence = instance.client.run()
    acknowledged = instance.client.wait_for_ack(sequence)
    if acknowledged.ack_seq != sequence:
        instance.close()
        raise RuntimeError("run command was not acknowledged")
    try:
        while time.monotonic() - started < timeout:
            frames = instance.client.drain_frames()
            states.extend(copy_state(frame) for frame in frames if frame.frameId > states[-1].frameId)
            if not instance.process.is_running():
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(f"replay playback exceeded {timeout:.0f}s")

        try:
            exit_code = instance.process.wait(timeout=5.0)
        except psutil.TimeoutExpired:
            raise RuntimeError("ReplayDnD did not auto-shutdown after replay completion")
        if not states:
            raise RuntimeError("replay produced no frame states")
        frame_ids = [state.frameId for state in states]
        missing = [
            expected for expected, actual in enumerate(frame_ids)
            if expected != actual
        ]
        snapshot = instance.client.snapshot()
        metadata = {
            "pid": pid,
            "frames": len(states),
            "last_frame": states[-1].frameId,
            "dropped_frames": snapshot.dropped_frames,
            "first_non_contiguous_index": missing[0] if missing else None,
            "exit_code": exit_code,
        }
        return states, metadata
    finally:
        instance.client.close()
        if instance.process.is_running():
            sokurl.shutdown(5.0, pid)


def select_complex_targets(states: list[RawFrameState]) -> list[int]:
    if len(states) < 2:
        raise RuntimeError("replay trace is too short")

    def score(index: int) -> int:
        state = states[index]
        previous = states[index - 1]
        hp_delta = max(0, previous.p1.hp - state.p1.hp) + max(0, previous.p2.hp - state.p2.hp)
        objects = state.p1ObjectCount + state.p2ObjectCount
        airborne = state.p1.airborne + state.p2.airborne
        hitstop = state.p1.hitstop + state.p2.hitstop
        untech = state.p1.untech + state.p2.untech
        return objects * 100 + min(hp_delta, 2000) * 4 + airborne * 50 + hitstop * 20 + min(untech, 100)

    candidates = range(1, len(states))
    peak_objects = max(candidates, key=lambda index: states[index].p1ObjectCount + states[index].p2ObjectCount)
    peak_complexity = max(candidates, key=score)
    airborne_damage = max(
        candidates,
        key=lambda index: (
            (states[index].p1.airborne + states[index].p2.airborne) * 10000
            + max(0, states[index - 1].p1.hp - states[index].p1.hp)
            + max(0, states[index - 1].p2.hp - states[index].p2.hp)
            + states[index].p1.hitstop + states[index].p2.hitstop
        ),
    )
    return list(dict.fromkeys((peak_objects, peak_complexity, airborne_damage)))


def replay_to_target(
    replay: Path,
    recorded: list[RawFrameState],
    inputs: list[InputPair],
    target: int,
    launch_timeout: float,
) -> tuple[RawFrameState, int | None, list[dict[str, object]]]:
    # A paused replay advances only on an acknowledged one-frame command.
    # It cannot overflow the frame ring, so this process needs no wall-clock cap.
    instance = launch_replay_checkpoint(replay, launch_timeout, True)
    try:
        initial_raw = instance.state
        differences = complex_state_diff(recorded[0], initial_raw)
        if differences:
            return initial_raw, 0, differences
        current = instance.apply_simple(recorded[0])
        differences = state_diff(recorded[0], current)
        if differences:
            return current, 0, differences

        if len(inputs) < target:
            raise ValueError("recorded input trace is shorter than the reconstruction target")
        for frame in range(1, target + 1):
            simulated = instance.step_native()
            differences = complex_state_diff(recorded[frame], simulated)
            if differences:
                return simulated, frame, differences
            current = instance.apply_simple(recorded[frame])
            differences = state_diff(recorded[frame], current)
            if differences:
                return current, frame, differences
        return current, None, []
    finally:
        instance.close()


def summarize_frame(state: RawFrameState) -> dict[str, object]:
    return {
        "frame": state.frameId,
        "hash": f"{state.stateHash:016X}",
        "p1": {
            "character": state.p1.characterId,
            "hp": state.p1.hp,
            "action": state.p1.actionId,
            "sequence": state.p1.sequenceId,
            "subsequence": state.p1.subsequenceId,
            "airborne": state.p1.airborne,
            "hitstop": state.p1.hitstop,
            "objects": state.p1ObjectCount,
        },
        "p2": {
            "character": state.p2.characterId,
            "hp": state.p2.hp,
            "action": state.p2.actionId,
            "sequence": state.p2.sequenceId,
            "subsequence": state.p2.subsequenceId,
            "airborne": state.p2.airborne,
            "hitstop": state.p2.hitstop,
            "objects": state.p2ObjectCount,
        },
    }


def run_validation(replay: Path, launch_timeout: float, playback_timeout: float) -> dict[str, object]:
    states, playback = play_full_replay(replay, launch_timeout, playback_timeout)
    inputs = [
        InputPair(input_tuple(state.p1.input), input_tuple(state.p2.input))
        for state in states[1:]
    ]
    targets = select_complex_targets(states)
    report: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "replay": str(replay.resolve()),
        "playback": playback,
        "characters": [states[0].p1.characterId, states[0].p2.characterId],
        "max_objects": [
            max(state.p1ObjectCount for state in states),
            max(state.p2ObjectCount for state in states),
        ],
        "airborne_frames": [
            sum(bool(state.p1.airborne) for state in states),
            sum(bool(state.p2.airborne) for state in states),
        ],
        "hitstop_frames": [
            sum(bool(state.p1.hitstop) for state in states),
            sum(bool(state.p2.hitstop) for state in states),
        ],
        "targets": [summarize_frame(states[target]) for target in targets],
        "target_results": [],
    }
    for target in targets:
        reconstructed, divergence, differences = replay_to_target(replay, states, inputs, target, launch_timeout)
        result = {
            "target": target,
            "expected_hash": f"{states[target].stateHash:016X}",
            "actual_hash": f"{reconstructed.stateHash:016X}",
            "first_divergent_frame": divergence,
            "differences": differences[:100],
        }
        report["target_results"].append(result)  # type: ignore[union-attr]
        if divergence is not None:
            report["success"] = False
            return report
    report["success"] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReplayDnD playback and reconstruction validator")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_validation(args.replay, 35.0, 900.0)
    except (BridgeUnavailable, OSError, psutil.Error, RuntimeError, ValueError) as error:
        report = {"success": False, "error": str(error), "replay": str(args.replay)}
    path = args.report
    if path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPORT_DIR / f"replay-validation-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
