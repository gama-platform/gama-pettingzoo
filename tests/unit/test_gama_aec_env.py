"""
Unit tests for GAMA AEC Environment.

This module tests the core functionality of the GamaAECEnv class
for multi-agent Agent-Environment-Cycle execution.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pettingzoo import AECEnv

# Import the class to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gama_pettingzoo.gama_aec_env import GamaAECEnv


class TestGamaAECEnvInit:
    """Test GamaAECEnv initialization."""
    
    def test_init_with_valid_config(self, sample_agent_configs):
        """Test initialization with valid configuration."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            assert env.agents == config["agents_list"]
            assert env.possible_agents == config["agents_list"]
            assert env.num_agents == len(config["agents_list"])
            assert env.experiment_name == "test_experiment"
            assert env.agent_selection == config["agents_list"][0]  # First agent should be selected
    
    def test_init_invalid_agents_list(self):
        """Test initialization with invalid agents list."""
        with pytest.raises(ValueError, match="agents_list must be a non-empty list"):
            GamaAECEnv(
                experiment_name="test",
                agents_list=[],
                action_spaces={},
                observation_spaces={}
            )
    
    def test_init_mismatched_spaces(self):
        """Test initialization with mismatched action and observation spaces."""
        with pytest.raises(ValueError):
            GamaAECEnv(
                experiment_name="test",
                agents_list=["agent1", "agent2"],
                action_spaces={"agent1": {"type": "Discrete", "n": 4}},
                observation_spaces={"agent1": {"type": "Box", "low": 0, "high": 1, "shape": [2], "dtype": "float"}}
            )


class TestGamaAECEnvReset:
    """Test GamaAECEnv reset functionality."""
    
    @pytest.fixture
    def mock_env(self, sample_agent_configs):
        """Create a mock environment for testing."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            return env, mock_client
    
    def test_reset_successful(self, mock_env, sample_gama_multiagent_data):
        """Test successful reset operation."""
        env, mock_client = mock_env
        
        # Mock GAMA response for reset
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str(sample_gama_multiagent_data["prison_escape_step"])
        }
        
        env.reset()
        
        # Check that environment is properly reset
        assert env.agent_selection == env.agents[0]
        assert len(env.agents) == env.num_agents
        
        # Check that observations are available
        observation = env.observe(env.agent_selection)
        assert observation is not None
    
    def test_reset_with_seed(self, mock_env, sample_gama_multiagent_data):
        """Test reset with seed parameter."""
        env, mock_client = mock_env
        
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str(sample_gama_multiagent_data["prison_escape_step"])
        }
        
        env.reset(seed=42)
        
        assert env.agent_selection == env.agents[0]
    
    def test_reset_failure(self, mock_env):
        """Test reset when GAMA returns error."""
        env, mock_client = mock_env
        
        mock_client.send_message.return_value = {
            "type": "Error",
            "content": "Reset failed"
        }
        
        with pytest.raises(Exception):
            env.reset()


class TestGamaAECEnvStep:
    """Test GamaAECEnv step functionality."""
    
    @pytest.fixture
    def initialized_env(self, sample_agent_configs, sample_gama_multiagent_data):
        """Create an initialized environment for step testing."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock successful reset
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset()
            return env, mock_client
    
    def test_step_successful(self, initialized_env, sample_gama_multiagent_data):
        """Test successful step operation."""
        env, mock_client = initialized_env
        current_agent = env.agent_selection
        
        # Mock GAMA response for step
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str(sample_gama_multiagent_data["prison_escape_step"])
        }
        
        # Take a step with the current agent
        env.step(0)  # Action 0 (stay)
        
        # Agent selection should change after step (in AEC, agents take turns)
        assert env.agent_selection != current_agent or len(env.agents) == 1
    
    def test_step_terminal_state(self, initialized_env, sample_gama_multiagent_data):
        """Test step that results in terminal state."""
        env, mock_client = initialized_env
        
        # Mock terminal state response
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str(sample_gama_multiagent_data["terminal_step"])
        }
        
        env.step(0)
        
        # Check that environment detects terminal state
        observation, reward, termination, truncation, info = env.last()
        assert termination is True
    
    def test_step_failure(self, initialized_env):
        """Test step when GAMA returns error."""
        env, mock_client = initialized_env
        
        mock_client.send_message.return_value = {
            "type": "Error",
            "content": "Step failed"
        }
        
        with pytest.raises(Exception):
            env.step(0)


class TestGamaAECEnvObserve:
    """Test GamaAECEnv observation functionality."""
    
    @pytest.fixture
    def initialized_env(self, sample_agent_configs, sample_gama_multiagent_data):
        """Create an initialized environment for observation testing."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"], 
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock successful reset
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset()
            return env, mock_client
    
    def test_observe_current_agent(self, initialized_env):
        """Test observing the current agent."""
        env, _ = initialized_env
        
        observation = env.observe(env.agent_selection)
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
    
    def test_observe_specific_agent(self, initialized_env):
        """Test observing a specific agent."""
        env, _ = initialized_env
        
        for agent in env.agents:
            observation = env.observe(agent)
            assert observation is not None
            assert isinstance(observation, np.ndarray)
    
    def test_observe_invalid_agent(self, initialized_env):
        """Test observing an invalid agent."""
        env, _ = initialized_env
        
        with pytest.raises(ValueError):
            env.observe("invalid_agent")


class TestGamaAECEnvLast:
    """Test GamaAECEnv last() functionality."""
    
    @pytest.fixture
    def stepped_env(self, sample_agent_configs, sample_gama_multiagent_data):
        """Create an environment that has taken a step."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset()
            env.step(0)  # Take one step
            return env, mock_client
    
    def test_last_returns_tuple(self, stepped_env):
        """Test that last() returns the correct tuple."""
        env, _ = stepped_env
        
        result = env.last()
        
        assert isinstance(result, tuple)
        assert len(result) == 5  # observation, reward, termination, truncation, info
        
        observation, reward, termination, truncation, info = result
        
        assert observation is not None
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(termination, bool)
        assert isinstance(truncation, bool)
        assert isinstance(info, dict)
    
    def test_last_observation_matches_observe(self, stepped_env):
        """Test that last() observation matches observe()."""
        env, _ = stepped_env
        
        observation_from_last, _, _, _, _ = env.last()
        observation_from_observe = env.observe(env.agent_selection)
        
        np.testing.assert_array_equal(observation_from_last, observation_from_observe)


class TestGamaAECEnvAgentSelection:
    """Test agent selection mechanism."""
    
    def test_agent_selection_cycling(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test that agent selection cycles through agents."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset()
            
            initial_agent = env.agent_selection
            
            # Take steps for all agents in a round
            agents_seen = []
            for _ in range(len(env.agents)):
                agents_seen.append(env.agent_selection)
                env.step(0)
            
            # Should have seen all agents
            assert len(set(agents_seen)) == len(env.agents)
    
    def test_agent_selection_after_reset(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test that agent selection resets properly."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset()
            initial_agent = env.agent_selection
            
            # Take some steps
            env.step(0)
            env.step(1)
            
            # Reset again
            env.reset()
            
            # Should be back to initial agent
            assert env.agent_selection == initial_agent


class TestGamaAECEnvSpaces:
    """Test space-related functionality."""
    
    def test_action_space_property(self, sample_agent_configs):
        """Test action_space property."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            assert hasattr(env, 'action_space')
    
    def test_observation_space_property(self, sample_agent_configs):
        """Test observation_space property."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            assert hasattr(env, 'observation_space')


class TestGamaAECEnvUtilities:
    """Test utility functions and properties."""
    
    def test_render_not_implemented(self, sample_agent_configs):
        """Test that render raises NotImplementedError."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            with pytest.raises(NotImplementedError):
                env.render()
    
    def test_close(self, sample_agent_configs):
        """Test environment close functionality."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            env.close()
            
            # Verify client close was called
            mock_client.close.assert_called_once()
    
    def test_state_not_implemented(self, sample_agent_configs):
        """Test that state property raises NotImplementedError."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            with pytest.raises(NotImplementedError):
                _ = env.state()


@pytest.mark.multiagent
class TestGamaAECEnvMultiAgent:
    """Test multi-agent specific functionality."""
    
    def test_aec_turn_based_behavior(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test that AEC environment follows turn-based behavior."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset()
            
            # Track agent turns
            agent_turns = []
            for _ in range(6):  # Take several steps
                agent_turns.append(env.agent_selection)
                env.step(0)
                
                if all(env.terminations.values()):  # If all agents terminated
                    break
            
            # Should alternate between agents (for 2-agent case)
            if len(config["agents_list"]) == 2:
                for i in range(1, len(agent_turns)):
                    assert agent_turns[i] != agent_turns[i-1]
    
    @pytest.mark.parametrize("num_agents", [2, 3, 4])
    def test_scalability_with_agent_count(self, num_agents):
        """Test AEC environment scalability with different agent counts."""
        agents_list = [f"agent_{i}" for i in range(num_agents)]
        action_spaces = {agent: {"type": "Discrete", "n": 4} for agent in agents_list}
        observation_spaces = {agent: {"type": "Box", "low": 0, "high": 1, "shape": [2], "dtype": "float"} for agent in agents_list}
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=agents_list,
                action_spaces=action_spaces,
                observation_spaces=observation_spaces
            )
            
            assert env.num_agents == num_agents
            assert len(env.agents) == num_agents
            assert len(env.possible_agents) == num_agents
            assert env.agent_selection in agents_list


@pytest.mark.integration
class TestGamaAECEnvIntegration:
    """Integration tests (these would normally require actual GAMA server)."""
    
    @pytest.mark.skip(reason="Requires actual GAMA server")
    def test_full_episode_integration(self):
        """Test a full episode with actual GAMA server."""
        # This test would be run with an actual GAMA server
        pass
    
    @pytest.mark.skip(reason="Requires actual GAMA server")
    def test_aec_vs_parallel_consistency(self):
        """Test that AEC and Parallel environments produce consistent results."""
        # This test would verify that AEC and Parallel modes give equivalent outcomes
        pass


@pytest.mark.performance
class TestGamaAECEnvPerformance:
    """Performance tests for the AEC environment."""
    
    def test_step_performance(self, sample_agent_configs, performance_benchmark):
        """Test step operation performance."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str({
                    "States": {"prisoner": [0, 1, 3, 3, 6, 6], "guard": [3, 3, 0, 1, 6, 6]},
                    "Rewards": {"prisoner": 0.0, "guard": 0.0},
                    "Terminated": {"prisoner": False, "guard": False},
                    "Truncated": {"prisoner": False, "guard": False},
                    "Infos": {"prisoner": {}, "guard": {}},
                    "Agents": ["prisoner", "guard"]
                })
            }
            
            env.reset()
            
            with performance_benchmark.time_operation("aec_step"):
                env.step(0)
            
            # AEC step should complete within reasonable time
            performance_benchmark.assert_performance("aec_step", max_time=1.0)
    
    def test_observe_performance(self, sample_agent_configs, performance_benchmark):
        """Test observation operation performance."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str({
                    "States": {"prisoner": [0, 0, 3, 3, 6, 6], "guard": [3, 3, 0, 0, 6, 6]},
                    "Infos": {"prisoner": {}, "guard": {}},
                    "Agents": ["prisoner", "guard"]
                })
            }
            
            env.reset()
            
            with performance_benchmark.time_operation("observe"):
                env.observe(env.agent_selection)
            
            # Observation should be very fast
            performance_benchmark.assert_performance("observe", max_time=0.1)