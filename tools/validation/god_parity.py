"""Compare original and migrated script inputs for complete real game episodes."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
from game_batch import SokuGameBatch
import sokurl
from god_reference.compare import players_equal
from god_reference.reader import Reference
from god_reference.scheduler import OriginalScheduler
from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import decode_action
from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.runtime import GodPolicy


def episode(package, script, seat, config, reference, output):
    selections = [PlayerSetup(config.validation.opponent_character, 0, 0)] * 2
    selections[seat] = PlayerSetup(int(script[:2]), seat, seat % 4)
    episode_config = replace(EpisodeConfig.from_dict(OmegaConf.to_container(config.episode)),
                             match=MatchConfig(*selections))
    game = SokuGameBatch(float(config.runtime.launch_timeout))
    game.configure_observation(episode_config.backend_observation())
    originals = []
    result = {"script": script, "seat": seat, "success": False, "compared_frames": 0,
              "episode": OmegaConf.to_container(OmegaConf.structured(episode_config))}
    started = time.monotonic()
    try:
        current = game.reset_slots({0: int(config.seed)})[0]
        reader = game.privileged_readers[0]
        reference.initialize(reader.memory)
        scripts = [package.character_script(player.character) for player in selections]
        scripts[seat] = script
        actors = [GodPolicy(name, package, name, episode_config).spawn(config.seed + index)
                  for index, name in enumerate(scripts)]
        originals = [OriginalScheduler(package, name, config.seed + index, current.observations[index])
                     for index, name in enumerate(scripts)]
        result["decks"] = [p["deck"] for p in current.observations[0].players]
        with gzip.open(output / "frames.jsonl.gz", "wt", encoding="utf-8") as trace:
            while not current.ended and current.frame < episode_config.max_frames:
                if current.frame % 60 == 0 and Path(config.validation.stop_file).is_file():
                    raise InterruptedError("validation stop requested")
                reference.dll.reference_reload(reader.value(0x8985E4, "I"), 1,
                                               current.observations[0].world["weather"])
                players_equal(reference, current.observations[0].players)
                actions = []
                for index, actor in enumerate(actors):
                    value = current.observations[index]
                    if current.frame:
                        originals[index].advance(value)
                    inputs = decode_action(actor.act(episode_config.encode(value)))
                    expected = originals[index].inputs()
                    if actor.failures or inputs.inputs != expected:
                        raise AssertionError((current.frame, index, inputs.inputs, expected, actor.failures))
                    actions.append(inputs)
                trace.write(json.dumps({"frame": current.frame, "inputs": [a.inputs for a in actions],
                                        "state": dict(current.diagnostics)}) + "\n")
                result["compared_frames"] += 1
                current = game.step({0: tuple(actions)})[0]
                if current.frame % 600 == 0:
                    print(script, seat, "frame", current.frame, "hp", current.diagnostics["hp"], flush=True)
        result.update(success=True, outcome=current.outcome.value if current.ended else "time_limit",
                      final=dict(current.diagnostics))
    except Exception as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        if "actors" in locals():
            result["script_failures"] = [[x.decode("utf-8", errors="replace") if isinstance(x, bytes)
                                           else repr(x) for x in a.failures] for a in actors]
    finally:
        for original in originals:
            original.close()
        game.close()
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@hydra.main(version_base="1.3", config_path="../../config", config_name="god_parity")
def main(config):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sokurl.GAME_DIR = Path(config.validation.game_directory).resolve(strict=True)
    sokurl.GAME_EXE = sokurl.GAME_DIR / "th123.exe"
    sokurl.SKIPINTRO_INI = sokurl.GAME_DIR / "modules/SkipIntro/SkipIntro.ini"
    package = ScriptPackage(config.rules.god.package, config.rules.god.api_source)
    reference = Reference(Path(config.validation.reference_library).resolve(strict=True))
    names = (sorted(name for name in package.files
                    if name.endswith(".ai") and name[:2].isdigit() and "_main" in name)
             if config.validation.scripts == "all" else list(config.validation.scripts))
    if not 0 <= config.validation.shard < config.validation.shards <= len(names):
        raise ValueError("validation requires 0 <= shard < shards <= script count")
    names = names[config.validation.shard::config.validation.shards]
    report = {"package": package.fingerprint, "original_package": package.original_fingerprint,
              "repairs": {name: value.decode("ascii") for name, value in package.repairs.items()},
              "reader_repairs": ["Initialize new object floats to zero before the original read filter.",
                                 "Use packaged 0.93 collision conversion order and hit-box flags."],
              "reference_sha256": hashlib.sha256(Path(config.validation.reference_library).read_bytes()).hexdigest(),
              "bridge_sha256": hashlib.sha256((sokurl.GAME_DIR / "modules/SokuRLBridge/SokuRLBridge.dll").read_bytes()).hexdigest(),
              "results": []}
    OmegaConf.save(config, output / "config.yaml")
    for name in names:
        for seat in config.validation.seats:
            if Path(config.validation.stop_file).is_file():
                raise InterruptedError("validation stop requested; completed reports are preserved")
            case = output / f"{Path(name).stem}-seat{seat}"
            case.mkdir()
            result = episode(package, name, seat, config, reference, case)
            report["results"].append(result)
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=True), flush=True)
    if not all(item["success"] for item in report["results"]):
        raise RuntimeError("original strategy parity failed; inspect the saved per-case evidence")


if __name__ == "__main__":
    main()
