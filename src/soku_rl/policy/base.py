"""Policy definitions create independent actors for each player and episode."""
from abc import ABC, abstractmethod
from typing import Protocol


class Actor(Protocol):
    def act(self, observation): ...


class Policy(ABC):
    name: str
    fingerprint: str

    @abstractmethod
    def spawn(self, seed: int) -> Actor:
        """Create fresh memory and a private random stream for one episode."""


class RLPolicy(Policy):
    """A policy whose action distribution is represented by learned parameters."""


class RulePolicy(Policy):
    """A policy whose actions are produced by explicit rules or source scripts."""
