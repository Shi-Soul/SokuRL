"""Batch stateless greedy DQN inference while retaining other actors' call order."""
import numpy as np

from soku_rl.policy.dqn import DQNEpisode


def episode_actions(requests):
    actions, groups = {}, {}
    for key, (actor, observation) in requests.items():
        if isinstance(actor, DQNEpisode):
            groups.setdefault(id(actor.model), []).append((key, actor, observation))
        else:
            actions[key] = actor.act(observation)
    for entries in groups.values():
        observations = [entry[2] for entry in entries]
        if isinstance(observations[0], dict):
            batch = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
        else:
            batch = np.stack(observations)
        predicted, _ = entries[0][1].model.predict(batch, deterministic=True)
        for (key, _, _), action in zip(entries, predicted, strict=True):
            actions[key] = int(action)
    return actions
