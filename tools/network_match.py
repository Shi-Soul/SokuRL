"""Track original game rounds and complete matches without resetting the engine."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchEvent:
    kind: str
    match: int
    round: int
    scores: tuple[int, int]


class NetworkMatch:
    """Consume ordered snapshots. Scores decide results; zero HP only stops input.

    `round_started` tells the caller to reset observation history and policy memory.
    `match_finished` is emitted once when the configured number of wins is reached.
    Leaving network scenes without that result emits `match_interrupted`, not a loss.
    This observer does not control the game's rules, menus, or network connection.
    """

    def __init__(self, wins_required):
        if type(wins_required) is not int or wins_required < 1:
            raise ValueError("wins_required must be a positive integer")
        self.wins_required = wins_required
        self.phase = "waiting"
        self._match = 0
        self._round = -1
        self._frame = 0
        self._scores = (0, 0)
        self._active = False
        self._finished = False

    @property
    def can_act(self):
        return self.phase == "battle"

    def update(self, snapshot):
        events = []

        def emit(kind):
            events.append(MatchEvent(kind, self._match, self._round, self._scores))

        if snapshot.in_battle and snapshot.match != self._match:
            if snapshot.match < self._match:
                raise ValueError("network match counter moved backwards")
            if self._active and not self._finished:
                emit("match_interrupted")
            self._match, self._round = snapshot.match, -1
            self._frame, self._scores = 0, (0, 0)
            self._active, self._finished = True, False
            emit("match_started")

        if self._active and snapshot.match == self._match:
            if snapshot.updates < self._frame:
                raise ValueError("network update counter moved backwards")
            if any(new < old for new, old in zip(snapshot.scores, self._scores)):
                raise ValueError("round score decreased within a match")
            self._frame = snapshot.updates
            if snapshot.scores != self._scores:
                self._scores = snapshot.scores
                emit("score_changed")
            if not self._finished and max(self._scores) >= self.wins_required:
                self._finished = True
                emit("match_finished")

        network_scene = snapshot.connected and snapshot.scene in (8, 9, 10, 11, 13, 14)
        if self._active and not snapshot.in_battle:
            if not self._finished:
                emit("match_interrupted")
            self._active = False
        if not network_scene:
            self.phase = "disconnected"
        elif not snapshot.in_battle:
            self.phase = "menu" if snapshot.scene in (8, 9) else "loading"
        elif self._finished:
            self.phase = "match_finished"
        elif min(snapshot.raw.p1.hp, snapshot.raw.p2.hp) <= 0:
            self.phase = "between_rounds"
        else:
            if snapshot.raw.roundId != self._round:
                if snapshot.raw.roundId < self._round:
                    raise ValueError("round counter moved backwards")
                self._round = snapshot.raw.roundId
                emit("round_started")
            self.phase = "battle"
        return tuple(events)
