import json
from typing import Any, Dict

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes

from gama_gymnasium import GamaClientWrapper, GamaCommandError


class GamaClientWrapperPtZ(GamaClientWrapper):
    """
    GAMA client wrapper for PettingZoo environments.
    """
    def __init__(self, ip_address: str = None, port: int = 6868):
        super().__init__(ip_address=ip_address, port=port)
        
    def get_agents(self, experiment_id):
        return self._execute_expression(
            experiment_id,
            r"PetzAgent[0].agents"
        )
        
    def get_possible_agents(self, experiment_id):
        return self._execute_expression(
            experiment_id,
            r"PetzAgent[0].possible_agents"
        )

    def get_observation_spaces(self, experiment_id):
        return self._execute_expression(
            experiment_id, 
            r"PetzAgent[0].observation_spaces"
        )
    
    def get_action_spaces(self, experiment_id):
        return self._execute_expression(
            experiment_id, 
            r"PetzAgent[0].action_spaces"
        )
    def get_observations(self, experiment_id: str) -> Dict:
        """Get current observations from GAMA."""
        return self._execute_expression(experiment_id, r"PetzAgent[0].observations")

    def get_infos(self, experiment_id: str) -> Dict:
        """Get current infos from GAMA."""
        return self._execute_expression(experiment_id, r"PetzAgent[0].infos")
    
    def execute_step(self, experiment_id: str, actions) -> Dict:
        # Set actions
        # actions = 17.3
        # print(f"Actions {actions} of type {type(actions)}")
        self._execute_expression(
            experiment_id, 
            f'PetzAgent[0].actions <- from_json(\'{actions}\');'
        )
        
        # Execute step
        response = self.client.step(experiment_id, sync=True)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"Failed to execute step: {response}")
        
        # Get results
        data = self._execute_expression(experiment_id, r"PetzAgent[0].data")
        # print(f"Step data: {data}")
        return data