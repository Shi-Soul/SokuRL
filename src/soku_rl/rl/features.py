"""Encode complete fighter and ordered object records for the shared PPO."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH,
    PRIVILEGED_FEATURES, WORLD_NAMES)


class PrivilegedFeatures(BaseFeaturesExtractor):
    numeric_width = 2

    def __init__(self, observation_space, history_frames, object_features, player_features, features_dim):
        if min(history_frames, object_features, player_features, features_dim) < 1:
            raise ValueError("feature dimensions must be positive")
        width = observation_space.shape
        if len(width) != 1 or width[0] < history_frames * PRIVILEGED_FEATURES:
            raise ValueError("complete privileged observations are required")
        super().__init__(observation_space, features_dim)
        self.history_frames = history_frames
        self.base_width = history_frames * PRIVILEGED_FEATURES
        self.object_encoder = nn.Sequential(nn.Linear(OBJECT_WIDTH * self.numeric_width, object_features), nn.Tanh())
        self.player_encoder = nn.Sequential(
            nn.Linear(FIGHTER_WIDTH * self.numeric_width + MAX_OBJECTS * object_features, player_features), nn.Tanh())
        width = history_frames * (len(WORLD_NAMES) * self.numeric_width + player_features * 2) + width[0] - self.base_width
        self.output = nn.Sequential(nn.Linear(width, features_dim), nn.Tanh())
        self.register_buffer("object_indices", torch.arange(MAX_OBJECTS), persistent=False)

    def numeric_features(self, records):
        return records

    def forward(self, observations):
        batch = observations.shape[0]
        frames = observations[:, :self.base_width].reshape(batch * self.history_frames, -1)
        world = frames[:, :len(WORLD_NAMES) * 2]
        players = frames[:, len(WORLD_NAMES) * 2:].reshape(-1, PLAYER_WIDTH * 2)
        fighters = players[:, :FIGHTER_WIDTH * 2]
        objects = players[:, FIGHTER_WIDTH * 2:].reshape(-1, MAX_OBJECTS, OBJECT_WIDTH * 2)
        count_index = FIGHTER_NAMES.index("obj_n") * 2
        counts = fighters[:, count_index] * 4294967296. + fighters[:, count_index + 1] * 65536.
        present = self.object_indices.unsqueeze(0) < counts.unsqueeze(1)
        # Keep list order: flatten the encoded records, without sorting,
        # selecting nearest objects, or averaging different objects together.
        encoded = self.object_encoder(self.numeric_features(objects)) * present.unsqueeze(-1)
        players = self.player_encoder(torch.cat((self.numeric_features(fighters), encoded.flatten(1)), dim=1))
        frames = torch.cat((self.numeric_features(world), players.reshape(batch * self.history_frames, -1)), dim=1)
        return self.output(torch.cat((frames.reshape(batch, -1), observations[:, self.base_width:]), dim=1))


class NumericPrivilegedFeatures(PrivilegedFeatures):
    """Append scaled numeric values while retaining every lossless input part."""
    numeric_width = 3

    def numeric_features(self, records):
        pairs = records.reshape(*records.shape[:-1], -1, 2)
        values = pairs[..., 0] * 4294967296. + pairs[..., 1] * 65536.
        scaled = values.sign() * torch.log1p(values.abs()) / 16.
        return torch.cat((records, scaled), dim=-1)
