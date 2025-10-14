"""
Unit tests for GAMA Parallel Environment.

This module tests the core functionality of the GamaParallelEnv class
for multi-agent parallel execution.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pettingzoo import ParallelEnv

# Import the class to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv


class TestGamaParallelEnvInit:
    """Test GamaParallelEnv initialization."""
    
    def test_init_with_valid_config(self, sample_agent_configs):
        """Test initialization with valid configuration."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
                experiment_name="test_experiment",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            assert env.agents == config["agents_list"]
            assert env.possible_agents == config["agents_list"]
            assert env.num_agents == len(config["agents_list"])
            assert env.experiment_name == "test_experiment"
    
    def test_init_invalid_agents_list(self):
        """Test initialization with invalid agents list."""
        with pytest.raises(ValueError, match="agents_list must be a non-empty list"):
            GamaParallelEnv(
                experiment_name="test",
                agents_list=[],
                action_spaces={},
                observation_spaces={}
            )
    
    def test_init_mismatched_spaces(self):
        """Test initialization with mismatched action and observation spaces."""
        with pytest.raises(ValueError):
            GamaParallelEnv(
                experiment_name="test",
                agents_list=["agent1", "agent2"],
                action_spaces={"agent1": {"type": "Discrete", "n": 4}},
                observation_spaces={"agent1": {"type": "Box", "low": 0, "high": 1, "shape": [2], "dtype": "float"}}
            )


class TestGamaParallelEnvReset:
    """Test GamaParallelEnv reset functionality."""
    
    @pytest.fixture
    def mock_env(self, sample_agent_configs):
        """Create a mock environment for testing."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
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
        
        observations, infos = env.reset()
        
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        assert len(observations) == env.num_agents
        assert len(infos) == env.num_agents
        
        for agent in env.agents:
            assert agent in observations
            assert agent in infos
            assert observations[agent] is not None
    
    def test_reset_with_seed(self, mock_env):
        """Test reset with seed parameter."""
        env, mock_client = mock_env
        
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str({"States": {"prisoner": [0, 0, 3, 3, 6, 6], "guard": [3, 3, 0, 0, 6, 6]}, 
                           "Infos": {"prisoner": {}, "guard": {}}, "Agents": ["prisoner", "guard"]})
        }
        
        observations, infos = env.reset(seed=42)
        
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
    
    def test_reset_failure(self, mock_env):
        """Test reset when GAMA returns error."""
        env, mock_client = mock_env
        
        mock_client.send_message.return_value = {
            "type": "Error",
            "content": "Reset failed"
        }
        
        with pytest.raises(Exception):
            env.reset()


class TestGamaParallelEnvStep:
    """Test GamaParallelEnv step functionality."""
    
    @pytest.fixture
    def initialized_env(self, sample_agent_configs, sample_gama_multiagent_data):
        """Create an initialized environment for step testing."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
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
    
    def test_step_successful(self, initialized_env, sample_multiagent_actions, sample_gama_multiagent_data):
        """Test successful step operation."""
        env, mock_client = initialized_env
        actions = sample_multiagent_actions["prison_escape_actions"]
        
        # Mock GAMA response for step
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str(sample_gama_multiagent_data["prison_escape_step"])
        }
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)
        
        for agent in env.agents:
            assert agent in observations
            assert agent in rewards
            assert agent in terminations
            assert agent in truncations
            assert agent in infos
    
    def test_step_with_invalid_actions(self, initialized_env):
        """Test step with invalid actions."""
        env, _ = initialized_env
        
        # Missing actions for some agents
        invalid_actions = {"prisoner": 0}  # Missing guard action
        
        with pytest.raises(ValueError, match="Action required for all agents"):
            env.step(invalid_actions)
    
    def test_step_terminal_state(self, initialized_env, sample_multiagent_actions, sample_gama_multiagent_data):
        """Test step that results in terminal state."""
        env, mock_client = initialized_env
        actions = sample_multiagent_actions["prison_escape_actions"]
        
        # Mock terminal state response
        mock_client.send_message.return_value = {
            "type": "CommandExecutedSuccessfully",
            "content": str(sample_gama_multiagent_data["terminal_step"])
        }
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check that all agents are terminated
        for agent in env.agents:
            assert terminations[agent] is True
    
    def test_step_failure(self, initialized_env, sample_multiagent_actions):
        """Test step when GAMA returns error."""
        env, mock_client = initialized_env
        actions = sample_multiagent_actions["prison_escape_actions"]
        
        mock_client.send_message.return_value = {
            "type": "Error",
            "content": "Step failed"
        }
        
        with pytest.raises(Exception):
            env.step(actions)


class TestGamaParallelEnvSpaces:
    """Test space-related functionality."""
    
    def test_action_space_property(self, sample_agent_configs):
        """Test action_space property."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            for agent in env.agents:
                assert hasattr(env, f'action_space')
                # The action space should be accessible for each agent
    
    def test_observation_space_property(self, sample_agent_configs):
        """Test observation_space property."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            for agent in env.agents:
                assert hasattr(env, f'observation_space')


class TestGamaParallelEnvUtilities:
    """Test utility functions and properties."""
    
    def test_render_not_implemented(self, sample_agent_configs):
        """Test that render raises NotImplementedError."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
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
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
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
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            with pytest.raises(NotImplementedError):
                _ = env.state()


@pytest.mark.multiagent
class TestGamaParallelEnvMultiAgent:
    """Test multi-agent specific functionality."""
    
    def test_multiple_agents_consistency(self, sample_agent_configs):
        """Test that multi-agent operations maintain consistency."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            assert len(env.agents) == len(env.possible_agents)
            assert env.num_agents == len(env.agents)
            assert all(agent in env.possible_agents for agent in env.agents)
    
    @pytest.mark.parametrize("num_agents", [2, 3, 4])
    def test_scalability_with_agent_count(self, num_agents):
        """Test environment scalability with different agent counts."""
        agents_list = [f"agent_{i}" for i in range(num_agents)]
        action_spaces = {agent: {"type": "Discrete", "n": 4} for agent in agents_list}
        observation_spaces = {agent: {"type": "Box", "low": 0, "high": 1, "shape": [2], "dtype": "float"} for agent in agents_list}
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'):
            env = GamaParallelEnv(
                experiment_name="test",
                agents_list=agents_list,
                action_spaces=action_spaces,
                observation_spaces=observation_spaces
            )
            
            assert env.num_agents == num_agents
            assert len(env.agents) == num_agents
            assert len(env.possible_agents) == num_agents


@pytest.mark.integration
class TestGamaParallelEnvIntegration:
    """Integration tests (these would normally require actual GAMA server)."""
    
    @pytest.mark.skip(reason="Requires actual GAMA server")
    def test_full_episode_integration(self):
        """Test a full episode with actual GAMA server."""
        # This test would be run with an actual GAMA server
        pass
    
    @pytest.mark.skip(reason="Requires actual GAMA server")
    def test_concurrent_environments(self):
        """Test multiple environments running concurrently."""
        # This test would verify multiple environments can run simultaneously
        pass


@pytest.mark.performance
class TestGamaParallelEnvPerformance:
    """Performance tests for the parallel environment."""
    
    def test_reset_performance(self, sample_agent_configs, performance_benchmark):
        """Test reset operation performance."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str({"States": {"prisoner": [0, 0, 3, 3, 6, 6], "guard": [3, 3, 0, 0, 6, 6]}, 
                               "Infos": {"prisoner": {}, "guard": {}}, "Agents": ["prisoner", "guard"]})
            }
            
            with performance_benchmark.time_operation("reset"):
                env.reset()
            
            # Reset should complete within reasonable time
            performance_benchmark.assert_performance("reset", max_time=1.0)
    
    def test_step_performance(self, sample_agent_configs, sample_multiagent_actions, performance_benchmark):
        """Test step operation performance."""
        config = sample_agent_configs["prison_escape"]
        actions = sample_multiagent_actions["prison_escape_actions"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
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
            
            # Initialize environment
            env.reset()
            
            with performance_benchmark.time_operation("step"):
                env.step(actions)
            
            # Step should complete within reasonable time
            performance_benchmark.assert_performance("step", max_time=1.0)