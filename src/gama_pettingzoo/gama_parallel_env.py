"""Simultaneous-action (Parallel) environment over a GAMA simulation."""

from typing import Any
import traceback
import json

from gama_gymnasium import GamaEnvironmentError

from pettingzoo import ParallelEnv

from .gama_env_base import GamaEnvBase


class GamaParallelEnv(GamaEnvBase, ParallelEnv):

    metadata = {"name": "GamaParallelEnv-v0"}

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str,
                 gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868,
                 render_mode=None):
        super().__init__(
            gaml_experiment_path=gaml_experiment_path,
            gaml_experiment_name=gaml_experiment_name,
            gaml_experiment_parameters=gaml_experiment_parameters,
            gama_ip_address=gama_ip_address,
            gama_port=gama_port,
            render_mode=render_mode,
        )

    def reset(self, seed: int = None, options: dict[str, Any] | None = None):
        """Reset the environment and return initial observation."""
        if seed is not None:
            self._seed(seed)

        # Reset GAMA experiment
        self.gama_client.reset_experiment(self.experiment_id, seed)

        self.agents = self.gama_client.get_agents(self.experiment_id)

        # Get initial state
        states = self.gama_client.get_observations(self.experiment_id)
        infos = self.gama_client.get_infos(self.experiment_id)

        observations = {}

        try:
            for agent, state in states.items():
                obs = self.space_converter.convert_gama_to_gym_observation(
                    self.observation_space(agent), state)
                observations[agent] = obs
        except Exception as e:
            print(f"Conversion error: {e}")
            traceback.print_exc()
            raise GamaEnvironmentError(f"Failed to convert observation: {e}")

        return observations, infos

    def step(self, actions, nb_steps: int = 1):
        gama_actions = {}
        for agent, action in actions.items():
            gama_actions[agent] = self._fast_act_encode(agent, action)

        # Compact JSON (no spaces after separators)
        actions_json = json.dumps(gama_actions, separators=(',', ':'))
        step_data = self.gama_client.execute_step(
            self.experiment_id, actions_json, nb_steps=nb_steps)

        observations = {}

        try:
            for agent, state in step_data['Observations'].items():
                observations[agent] = self._fast_obs_convert(agent, state)
        except Exception as e:
            print(f"Step conversion error: {e}")
            traceback.print_exc()
            raise GamaEnvironmentError(f"Failed to convert step observation: {e}")

        rewards = step_data['Rewards']
        terminated = step_data['Terminations']
        truncated = step_data['Truncations']
        infos = step_data['Infos']

        return observations, rewards, terminated, truncated, infos
