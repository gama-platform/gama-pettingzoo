"""Shared plumbing for the GAMA-backed PettingZoo environments."""

from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding

from gama_gymnasium import SpaceConverter

from .gama_client_wrapper import GamaClientWrapperPtZ


class GamaEnvBase:
    """Connection, experiment loading, space discovery and value conversion.

    Deliberately not a PettingZoo environment on its own: it carries no notion of a turn, a
    step or an episode, which is exactly the part the two subclasses do not share.
    """

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str,
                 gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868,
                 render_mode=None):
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.experiment_parameters = gaml_experiment_parameters or []
        self.render_mode = render_mode

        self.agents: list[str] = []
        self.possible_agents: list[str] = []

        self.gama_client = GamaClientWrapperPtZ(gama_ip_address, gama_port)
        self.space_converter = SpaceConverter()

        # Spaces are fixed for the lifetime of a run, so they are fetched once rather than on
        # every observation. `_*_spaces_raw` keeps GAMA's own description, the caches keep the
        # converted Gymnasium objects.
        self._obs_space_cache: dict = {}
        self._act_space_cache: dict = {}
        self._obs_spaces_raw: dict | None = None
        self._act_spaces_raw: dict | None = None

        self.np_random = None

        self._setup_experiment()

    # ── setup ────────────────────────────────────────────────────────────────────

    def _setup_experiment(self):
        """Load the experiment and learn what the model exposes."""
        self.experiment_id = self.gama_client.load_experiment(
            self.gaml_file_path, self.experiment_name, self.experiment_parameters)

        self.possible_agents = list(self.gama_client.get_possible_agents(self.experiment_id))
        self._prime_space_caches()

    def _prime_space_caches(self):
        """Fetch every space in one round-trip instead of one per agent."""
        obs_raw = self.gama_client.get_observation_spaces(self.experiment_id)
        act_raw = self.gama_client.get_action_spaces(self.experiment_id)
        self._obs_spaces_raw = obs_raw
        self._act_spaces_raw = act_raw

        for agent in self.possible_agents:
            if agent in obs_raw:
                self._obs_space_cache[agent] = self.space_converter.map_to_space(obs_raw[agent])
            if agent in act_raw:
                self._act_space_cache[agent] = self.space_converter.map_to_space(act_raw[agent])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # ── spaces ───────────────────────────────────────────────────────────────────

    def observation_space(self, agent):
        """Cached observation space; falls back to a round-trip for an agent primed late."""
        if agent not in self._obs_space_cache:
            obs_raw = self.gama_client.get_observation_spaces(self.experiment_id)
            self._obs_space_cache[agent] = SpaceConverter().map_to_space(obs_raw[agent])
        return self._obs_space_cache[agent]

    def action_space(self, agent):
        """Cached action space; falls back to a round-trip for an agent primed late."""
        if agent not in self._act_space_cache:
            act_raw = self.gama_client.get_action_spaces(self.experiment_id)
            self._act_space_cache[agent] = SpaceConverter().map_to_space(act_raw[agent])
        return self._act_space_cache[agent]

    # ── conversion ───────────────────────────────────────────────────────────────
    #
    # These two keep their original names because they are called from outside the library:
    # StarfarmParallelEnv (and any other subclass in the wild) uses them directly.

    def _fast_act_encode(self, agent: str, action) -> int | float | list:
        """Encode an action for GAMA without Gymnasium's `to_jsonable()` overhead.

        Discrete -> plain int, Box -> plain float or list of float; anything else falls back
        to the generic path.
        """
        space = self.action_space(agent)
        if isinstance(space, Discrete):
            return int(action)
        if isinstance(space, Box):
            if np.ndim(action) == 0:
                return float(action)
            return [float(v) for v in np.ravel(action)]
        return space.to_jsonable([action])[0]

    def _fast_obs_convert(self, agent: str, state) -> np.ndarray:
        """Convert a GAMA observation without `from_jsonable()` overhead."""
        space = self.observation_space(agent)
        if isinstance(space, Box):
            return np.array(state, dtype=space.dtype)
        return space.from_jsonable([state])[0]

    # ── misc ─────────────────────────────────────────────────────────────────────

    def render(self):
        pass

    def close(self):
        if getattr(self, "gama_client", None):
            try:
                self.gama_client.close()
            except Exception as e:
                print(f"Warning: Error closing GAMA environment: {e}")
            finally:
                self.gama_client = None
