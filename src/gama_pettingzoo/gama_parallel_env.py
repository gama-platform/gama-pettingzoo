from typing import Any
import traceback
import json

from gama_gymnasium import SpaceConverter, GamaEnvironmentError

from pettingzoo import ParallelEnv
from gymnasium.utils import seeding

from .gama_client_wrapper import GamaClientWrapperPtZ


class GamaParallelEnv(ParallelEnv):

    metadata = {"name": "GamaParallelEnv-v0"}

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str,
                 gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868,
                 render_mode = None):
        # Store configuration
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.experiment_parameters = gaml_experiment_parameters or []
        self.render_mode = render_mode
        
        self.agents = []
        self.possible_agents = []
        
        # Initialize GAMA client wrapper
        self.gama_client = GamaClientWrapperPtZ(gama_ip_address, gama_port)
        
        # Initialize space converter
        self.space_converter = SpaceConverter()
        
        # Connect and setup experiment
        self._setup_experiment()
        
    def _setup_experiment(self):
        """Setup the GAMA experiment and get experiment ID."""
        self.experiment_id = self.gama_client.load_experiment(
            self.gaml_file_path, 
            self.experiment_name, 
            self.experiment_parameters
        )
        
        self.possible_agents = self.gama_client.get_possible_agents(self.experiment_id)
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
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
                obs = self.space_converter.convert_gama_to_gym_observation(self.observation_space(agent), state)
                observations[agent] = obs
        except Exception as e:
            print(f"Conversion error: {e}")
            traceback.print_exc()
            raise GamaEnvironmentError(f"Failed to convert observation: {e}")
        
        return observations, infos
    
    def step(self, actions):
        gama_actions = {}
        for agent, action in actions.items():
            gama_actions[agent] = self.action_space(agent).to_jsonable([action])[0]
            
        step_data = self.gama_client.execute_step(self.experiment_id, json.dumps(gama_actions))
        
        observations = {}

        try:
            for agent, state in step_data['Observations'].items():
                obs = self.space_converter.convert_gama_to_gym_observation(self.observation_space(agent), state)
                observations[agent] = obs
        except Exception as e:
            print(f"Step conversion error: {e}")
            traceback.print_exc()
            raise GamaEnvironmentError(f"Failed to convert step observation: {e}")
        
        rewards = step_data['Rewards']
        terminated = step_data['Terminations']
        truncated = step_data['Truncations']
        infos = step_data['Infos']

        return observations, rewards, terminated, truncated, infos

    def render(self):
        pass
    
    def observation_space(self, agent):
        observation_spaces_data = self.gama_client.get_observation_spaces(self.experiment_id)
        return self.space_converter.map_to_space(observation_spaces_data[agent])

    def action_space(self, agent):
        action_spaces_data = self.gama_client.get_action_spaces(self.experiment_id)
        return self.space_converter.map_to_space(action_spaces_data[agent])

    def close(self):
        if hasattr(self, 'gama_client') and self.gama_client:
            try:
                self.gama_client.close()
            except Exception as e:
                print(f"Warning: Error closing GAMA environment: {e}")
            finally:
                self.gama_client = None