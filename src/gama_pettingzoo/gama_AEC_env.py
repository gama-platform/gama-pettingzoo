"""Turn-based (AEC) environment over a GAMA simulation."""

from typing import Any
import json
import traceback

import numpy as np
from gama_client.message_types import MessageTypes
from pettingzoo import AECEnv

try:
    # PettingZoo >= 1.24. `pettingzoo.utils.agent_selector` still resolves, but to the
    # MODULE, so importing the old lowercase name silently yields something uncallable.
    from pettingzoo.utils import AgentSelector
except ImportError:  # pragma: no cover - older PettingZoo
    from pettingzoo.utils import agent_selector as AgentSelector

from gama_gymnasium import GamaEnvironmentError

from .gama_env_base import GamaEnvBase


class GamaAECEnv(GamaEnvBase, AECEnv):
    """PettingZoo AEC environment backed by a GAMA experiment.

    Agents and their spaces are discovered from the running model, exactly as
    `GamaParallelEnv` does — the GAML model stays the single source of truth, so both
    environments can be pointed at the same experiment without duplicating declarations.
    """

    metadata = {"name": "GamaAECEnv-v0", "is_parallelizable": False}

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str,
                 gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868,
                 steps_per_round: int = 0, render_mode=None):
        # GAMA cycles to run once a round is complete. 0 for purely turn-based models, where
        # a round is the only thing that advances the world. See `_advance_simulation`.
        # Set before super().__init__, which loads the experiment straight away.
        self.steps_per_round = int(steps_per_round)

        self._agent_selector = None
        # Actions staged during the current round; kept for render/debugging only.
        self._staged_actions: dict[str, Any] = {}

        super().__init__(
            gaml_experiment_path=gaml_experiment_path,
            gaml_experiment_name=gaml_experiment_name,
            gaml_experiment_parameters=gaml_experiment_parameters,
            gama_ip_address=gama_ip_address,
            gama_port=gama_port,
            render_mode=render_mode,
        )

    def _setup_experiment(self):
        """Discover the model, then insist every agent is fully described.

        The parallel environment tolerates a missing space because it can fetch one lazily.
        Turn-based play cannot: `observe` and `step` are called per agent from the very first
        turn, so a gap surfaces as a bare KeyError mid-episode. Better to say so at load time.
        """
        super()._setup_experiment()

        if not self.possible_agents:
            raise GamaEnvironmentError(
                "The GAMA experiment reported no agents. Check that the model creates its "
                "PetzAgent and fills `possible_agents` during init.")

        missing = [a for a in self.possible_agents
                   if a not in self._obs_space_cache or a not in self._act_space_cache]
        if missing:
            raise GamaEnvironmentError(
                f"Missing observation or action space for agent(s): {missing}. Every agent in "
                "`possible_agents` needs an entry in both `observation_spaces` and "
                "`action_spaces`.")

    # ── episode lifecycle ────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed(seed)

        self.gama_client.reset_experiment(self.experiment_id, seed)

        self.agents = list(self.possible_agents)

        # Fixed turn order, restarted from the first agent on every reset.
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._staged_actions = {}

    def observe(self, agent: str) -> np.ndarray:
        """Current observation of `agent`, reflecting the actions already taken this round."""
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent {agent!r}. Known agents: {self.possible_agents}")

        state = self.gama_client._execute_expression(
            self.experiment_id, f'PetzAgent[0].observe_one("{agent}")')
        try:
            return self._fast_obs_convert(agent, state)
        except Exception as e:
            traceback.print_exc()
            raise GamaEnvironmentError(
                f"Failed to convert the observation of {agent!r}: {e}") from e

    def step(self, action):
        """Apply `action` for the agent whose turn it is, then hand the turn over.

        The simulation is advanced only once the round is complete, so the next agent in the
        round sees the consequences of this action without any simulated time passing.
        """
        agent = self.agent_selection

        # PettingZoo requires a terminated/truncated agent to be stepped with None; the base
        # class then removes it and selects the next one.
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Rewards are earned per round, not per turn: clearing here means the value reported
        # to this agent covers only what happens from its turn onward.
        self._clear_rewards()
        self._staged_actions[agent] = action

        encoded = json.dumps(self._fast_act_encode(agent, action), separators=(",", ":"))
        self.gama_client._execute_expression(
            self.experiment_id, f"PetzAgent[0].set_action_one(\"{agent}\", '{encoded}')")

        if self._agent_selector.is_last():
            self._advance_simulation()
            self._staged_actions.clear()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def _advance_simulation(self):
        """Close the round: let GAMA apply it, advance the clock, and publish the transition.

        `steps_per_round` decides whether simulated time passes at all. Turn-based domains
        (board games, auctions) have no clock of their own — the round IS the step, and 0 is
        right. Models with their own dynamics (a resource regrowing, a fire spreading) need
        the cycles, so `end_round()` stages the outcome and the GAMA cycles then run the
        reflexes that produce it.

        Isolated so a model with an irregular cadence — a round that ends only when the model
        says so — can override just this method.
        """
        self.gama_client._execute_expression(self.experiment_id, "PetzAgent[0].end_round()")

        if self.steps_per_round > 0:
            response = self.gama_client.client.step(
                self.experiment_id, nb_step=self.steps_per_round, sync=True)
            if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise GamaEnvironmentError(f"Failed to advance the simulation: {response}")

        self.gama_client._execute_expression(self.experiment_id, "PetzAgent[0].publish()")
        data = self.gama_client._execute_expression(self.experiment_id, "PetzAgent[0].data")

        try:
            self.rewards = {a: float(r) for a, r in data["Rewards"].items()}
            self.terminations = {a: bool(t) for a, t in data["Terminations"].items()}
            self.truncations = {a: bool(t) for a, t in data["Truncations"].items()}
            self.infos = dict(data["Infos"])
        except (KeyError, TypeError) as e:
            raise GamaEnvironmentError(
                f"PetzAgent[0].data is missing or malformed after end_round(): {e}. Expected "
                "keys Observations, Rewards, Terminations, Truncations, Infos.") from e

    def render(self):
        if self.render_mode is None:
            return
        print(f"turn: {self.agent_selection} | staged this round: {self._staged_actions}")
