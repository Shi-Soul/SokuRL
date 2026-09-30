"""Convert official replay files and environment rollout archives using Hydra."""
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.replay import Replay
from soku_rl.replay.rollout import export_replay, write_rollout


@hydra.main(version_base="1.3", config_path="../config", config_name="replay")
def main(config):
    source = Path(config.replay.source).resolve(strict=True)
    destination = Path(config.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if config.replay.operation == "export":
        export_replay(source, destination)
        print(f"回放已导出：{destination}")
    elif config.replay.operation == "import":
        import sokurl
        from game_runtime.trajectory import replay_frames
        sokurl.GAME_DIR = Path(config.replay.game_directory).resolve(strict=True)
        sokurl.GAME_EXE = sokurl.GAME_DIR / "th123.exe"
        sokurl.SKIPINTRO_INI = sokurl.GAME_DIR / "modules/SkipIntro/SkipIntro.ini"
        replay = Replay.decode(source.read_bytes())
        episode = EpisodeConfig.from_dict(OmegaConf.to_container(config.episode, resolve=True))
        index = config.replay.match_index
        frames = replay_frames(replay, index, episode, float(config.runtime.launch_timeout))
        try:
            count = write_rollout(destination, replay, index, episode, frames)
        finally:
            frames.close()
        print(f"已读取 {count} 个连续帧，轨迹文件：{destination}")
    else:
        raise ValueError("replay.operation must be import or export")


if __name__ == "__main__":
    main()
