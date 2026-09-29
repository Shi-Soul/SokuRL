"""Run one trained seat on consecutive public frames of an original game round."""
from soku_rl.env.encoding import AGENTS
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.env.wrappers.learning import LearningEpisode


class LivePolicy:
    """Keep model memory, public history and own commands local to one round.

    The caller supplies visibility-filtered observations and submits returned
    commands through the local input transport. Command history records issued
    intents, not proof of engine execution; the caller logs transport outcomes.
    Skipped decision windows do not invent actions or advance model memory.
    The training time-limit feature saturates, but the original match continues.
    """

    def __init__(self, policy, interface, seat):
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("live policy requires a local seat")
        if interface.episode.observation_mode not in ("state", "image"):
            raise ValueError("live policy requires public observations")
        self.policy, self.interface = policy, interface
        self.agent = AGENTS[seat]
        self.history = ObservationHistory(interface.episode)
        self.features = LearningEpisode(interface)
        self.active = False

    def start_round(self, frame, observations, seed):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("round seed must be a supported uint32")
        self.history.reset(frame, observations)
        self.features.reset_agent(self.agent, self.history.observations()[self.agent])
        self.actor = self.policy.spawn(seed)
        self.origin = self.next_decision = frame
        self.active = True

    @property
    def decision_due(self):
        return self.active and self.history.frame == self.next_decision

    def observe(self, frame, observations):
        if not self.active:
            raise RuntimeError("start a round before observing frames")
        if self.decision_due:
            raise RuntimeError("consume the pending decision before advancing history")
        self.history.append(frame, observations)

    def act(self):
        if not self.decision_due:
            raise RuntimeError("act exactly once at each decision boundary")
        base = self.history.observations()[self.agent]
        elapsed = min(self.history.frame-self.origin, self.interface.episode.max_frames)
        observation = self.features.observation(self.agent, base, elapsed)
        command = self.interface.command(self.actor.act(observation))
        self.features.record_command(self.agent, command)
        self.next_decision += self.interface.episode.decision_frames
        return command

    def skip_decision(self):
        if not self.decision_due:
            raise RuntimeError("skip only an unconsumed decision boundary")
        self.next_decision += self.interface.episode.decision_frames

    def stop(self):
        self.active = False
        self.history.clear()
