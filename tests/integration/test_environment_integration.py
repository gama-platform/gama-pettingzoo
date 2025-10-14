"""
Integration tests for GAMA-PettingZoo environments.

This module tests the integration between different components
and full workflow scenarios.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

# Import the classes to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from gama_pettingzoo.gama_parallel_env import GamaParallelEnv
    from gama_pettingzoo.gama_aec_env import GamaAECEnv
except ImportError:
    # Handle cases where modules might not be available
    GamaParallelEnv = None
    GamaAECEnv = None


@pytest.mark.integration
@pytest.mark.skipif(GamaParallelEnv is None or GamaAECEnv is None, 
                   reason="Environment classes not available")
class TestEnvironmentConsistency:
    """Test consistency between Parallel and AEC environments."""
    
    def test_parallel_vs_aec_initialization(self, sample_agent_configs):
        """Test that Parallel and AEC environments initialize consistently."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'), \
             patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            
            parallel_env = GamaParallelEnv(
                experiment_name="test_parallel",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            aec_env = GamaAECEnv(
                experiment_name="test_aec",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            # Both environments should have same basic properties
            assert parallel_env.agents == aec_env.agents
            assert parallel_env.possible_agents == aec_env.possible_agents
            assert parallel_env.num_agents == aec_env.num_agents
    
    def test_observation_space_consistency(self, sample_agent_configs):
        """Test that observation spaces are consistent between environments."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'), \
             patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            
            parallel_env = GamaParallelEnv(
                experiment_name="test_parallel",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            aec_env = GamaAECEnv(
                experiment_name="test_aec",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            # Observation spaces should be equivalent
            # Note: Actual comparison would depend on how spaces are implemented
            assert hasattr(parallel_env, 'observation_space')
            assert hasattr(aec_env, 'observation_space')
    
    def test_action_space_consistency(self, sample_agent_configs):
        """Test that action spaces are consistent between environments."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama'), \
             patch('gama_pettingzoo.gama_aec_env.connect_to_gama'):
            
            parallel_env = GamaParallelEnv(
                experiment_name="test_parallel",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            aec_env = GamaAECEnv(
                experiment_name="test_aec",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            # Action spaces should be equivalent
            assert hasattr(parallel_env, 'action_space')
            assert hasattr(aec_env, 'action_space')


@pytest.mark.integration
@pytest.mark.skipif(GamaParallelEnv is None, reason="GamaParallelEnv not available")
class TestFullWorkflowParallel:
    """Test full workflow scenarios for Parallel environment."""
    
    def test_complete_episode_simulation(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test a complete episode from start to finish."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="integration_test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock initial reset response
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            # Reset environment
            observations, infos = env.reset(seed=42)
            
            assert isinstance(observations, dict)
            assert isinstance(infos, dict)
            assert len(observations) == env.num_agents
            
            # Simulate episode steps
            episode_length = 0
            max_steps = 10
            
            while episode_length < max_steps:
                # Create actions for all agents
                actions = {}
                for agent in env.agents:
                    actions[agent] = np.random.randint(0, 4)  # Random action from discrete space
                
                # Mock step response
                step_data = sample_gama_multiagent_data["prison_escape_step"]
                if episode_length >= max_steps - 1:
                    # Terminal step
                    step_data = sample_gama_multiagent_data["terminal_step"]
                
                mock_client.send_message.return_value = {
                    "type": "CommandExecutedSuccessfully",
                    "content": str(step_data)
                }
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                assert isinstance(observations, dict)
                assert isinstance(rewards, dict)
                assert isinstance(terminations, dict)
                assert isinstance(truncations, dict)
                assert isinstance(infos, dict)
                
                episode_length += 1
                
                # Check if episode is done
                if all(terminations.values()) or all(truncations.values()):
                    break
            
            # Clean up
            env.close()
    
    def test_multiple_episodes(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test running multiple episodes consecutively."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="multi_episode_test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            num_episodes = 3
            episode_lengths = []
            
            for episode in range(num_episodes):
                # Mock reset response
                mock_client.send_message.return_value = {
                    "type": "CommandExecutedSuccessfully",
                    "content": str(sample_gama_multiagent_data["prison_escape_step"])
                }
                
                observations, infos = env.reset(seed=episode)
                
                episode_length = 0
                max_steps = 5
                
                while episode_length < max_steps:
                    actions = {agent: 0 for agent in env.agents}  # All agents stay
                    
                    # Mock step response (terminal after max_steps)
                    step_data = (sample_gama_multiagent_data["terminal_step"] 
                               if episode_length >= max_steps - 1 
                               else sample_gama_multiagent_data["prison_escape_step"])
                    
                    mock_client.send_message.return_value = {
                        "type": "CommandExecutedSuccessfully",
                        "content": str(step_data)
                    }
                    
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                    
                    episode_length += 1
                    
                    if all(terminations.values()) or all(truncations.values()):
                        break
                
                episode_lengths.append(episode_length)
            
            # All episodes should have run
            assert len(episode_lengths) == num_episodes
            assert all(length > 0 for length in episode_lengths)
            
            env.close()


@pytest.mark.integration
@pytest.mark.skipif(GamaAECEnv is None, reason="GamaAECEnv not available")
class TestFullWorkflowAEC:
    """Test full workflow scenarios for AEC environment."""
    
    def test_complete_episode_turn_based(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test a complete episode with turn-based execution."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="aec_integration_test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock reset response
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully", 
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            env.reset(seed=42)
            
            # Track agent turns
            agent_sequence = []
            step_count = 0
            max_steps = 10
            
            while step_count < max_steps:
                current_agent = env.agent_selection
                agent_sequence.append(current_agent)
                
                # Get observation for current agent
                observation = env.observe(current_agent)
                assert observation is not None
                
                # Mock step response
                step_data = (sample_gama_multiagent_data["terminal_step"] 
                           if step_count >= max_steps - 1 
                           else sample_gama_multiagent_data["prison_escape_step"])
                
                mock_client.send_message.return_value = {
                    "type": "CommandExecutedSuccessfully",
                    "content": str(step_data)
                }
                
                # Take action
                env.step(0)  # Stay action
                
                # Check last state
                observation, reward, termination, truncation, info = env.last()
                
                assert isinstance(reward, (int, float, np.number))
                assert isinstance(termination, bool)
                assert isinstance(truncation, bool)
                assert isinstance(info, dict)
                
                step_count += 1
                
                # Check if done
                if termination or truncation:
                    break
            
            # Verify turn-based behavior
            assert len(agent_sequence) > 0
            
            # For two agents, should alternate
            if len(env.agents) == 2:
                for i in range(1, min(len(agent_sequence), 4)):
                    assert agent_sequence[i] != agent_sequence[i-1]
            
            env.close()


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and recovery scenarios."""
    
    @pytest.mark.skipif(GamaParallelEnv is None, reason="GamaParallelEnv not available")
    def test_connection_error_recovery(self, sample_agent_configs):
        """Test handling of connection errors."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            # Simulate connection failure
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                env = GamaParallelEnv(
                    experiment_name="error_test",
                    agents_list=config["agents_list"],
                    action_spaces=config["action_spaces"],
                    observation_spaces=config["observation_spaces"]
                )
    
    @pytest.mark.skipif(GamaParallelEnv is None, reason="GamaParallelEnv not available")
    def test_gama_server_error_handling(self, sample_agent_configs):
        """Test handling of GAMA server errors during operation."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="server_error_test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock error response from GAMA
            mock_client.send_message.return_value = {
                "type": "Error",
                "content": "GAMA simulation error"
            }
            
            with pytest.raises(Exception):
                env.reset()
    
    @pytest.mark.skipif(GamaParallelEnv is None, reason="GamaParallelEnv not available")
    def test_invalid_action_handling(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test handling of invalid actions."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="invalid_action_test",
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
            
            # Test invalid actions
            invalid_actions = {
                "prisoner": 999,  # Invalid action value
                "guard": 0
            }
            
            # Should raise error for invalid actions
            # (This depends on implementation - might be validation in step() or space checking)
            with pytest.raises((ValueError, AssertionError)):
                env.step(invalid_actions)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance characteristics of full workflows."""
    
    @pytest.mark.skipif(GamaParallelEnv is None, reason="GamaParallelEnv not available")
    def test_parallel_environment_throughput(self, sample_agent_configs, sample_gama_multiagent_data, performance_benchmark):
        """Test throughput of parallel environment over many steps."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="throughput_test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock fast responses
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(sample_gama_multiagent_data["prison_escape_step"])
            }
            
            # Initialize
            env.reset()
            
            # Measure throughput
            num_steps = 100
            actions = {agent: 0 for agent in env.agents}
            
            with performance_benchmark.time_operation("parallel_throughput"):
                for _ in range(num_steps):
                    env.step(actions)
            
            # Calculate steps per second
            total_time = performance_benchmark.get_result("parallel_throughput")
            steps_per_second = num_steps / total_time
            
            # Should achieve reasonable throughput (this is with mocked GAMA)
            assert steps_per_second > 10, f"Throughput too low: {steps_per_second:.2f} steps/sec"
            
            env.close()
    
    @pytest.mark.skipif(GamaAECEnv is None, reason="GamaAECEnv not available")
    def test_aec_environment_latency(self, sample_agent_configs, sample_gama_multiagent_data, performance_benchmark):
        """Test latency characteristics of AEC environment."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_aec_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaAECEnv(
                experiment_name="latency_test",
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
            
            # Measure individual step latencies
            step_times = []
            num_measurements = 10
            
            for _ in range(num_measurements):
                start_time = time.time()
                env.step(0)
                end_time = time.time()
                step_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_latency = sum(step_times) / len(step_times)
            max_latency = max(step_times)
            
            # Latency should be consistent and low (with mocked GAMA)
            assert avg_latency < 0.1, f"Average latency too high: {avg_latency:.4f}s"
            assert max_latency < 0.2, f"Max latency too high: {max_latency:.4f}s"
            
            env.close()


@pytest.mark.integration
class TestDataFlow:
    """Test data flow and consistency across components."""
    
    @pytest.mark.skipif(GamaParallelEnv is None, reason="GamaParallelEnv not available")
    def test_observation_data_integrity(self, sample_agent_configs, sample_gama_multiagent_data):
        """Test that observation data maintains integrity through the pipeline."""
        config = sample_agent_configs["prison_escape"]
        
        with patch('gama_pettingzoo.gama_parallel_env.connect_to_gama') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            env = GamaParallelEnv(
                experiment_name="data_integrity_test",
                agents_list=config["agents_list"],
                action_spaces=config["action_spaces"],
                observation_spaces=config["observation_spaces"]
            )
            
            env.client = mock_client
            
            # Mock deterministic responses
            test_data = sample_gama_multiagent_data["prison_escape_step"]
            mock_client.send_message.return_value = {
                "type": "CommandExecutedSuccessfully",
                "content": str(test_data)
            }
            
            # Reset and get initial observations
            initial_obs, initial_infos = env.reset()
            
            # Verify observation structure
            assert isinstance(initial_obs, dict)
            assert len(initial_obs) == len(env.agents)
            
            for agent in env.agents:
                assert agent in initial_obs
                assert initial_obs[agent] is not None
                # Observations should be numpy arrays (depending on implementation)
                if isinstance(initial_obs[agent], np.ndarray):
                    assert initial_obs[agent].dtype in [np.int32, np.int64, np.float32, np.float64]
            
            # Take a step and verify data consistency
            actions = {agent: 0 for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Verify all return types
            assert isinstance(obs, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)
            
            # All dictionaries should have same keys (all agents)
            assert set(obs.keys()) == set(env.agents)
            assert set(rewards.keys()) == set(env.agents)
            assert set(terminations.keys()) == set(env.agents)
            assert set(truncations.keys()) == set(env.agents)
            assert set(infos.keys()) == set(env.agents)
            
            env.close()


@pytest.mark.integration
@pytest.mark.gama
class TestGAMAServerIntegration:
    """Integration tests that would require actual GAMA server."""
    
    @pytest.mark.skip(reason="Requires actual GAMA server connection")
    def test_real_gama_connection(self):
        """Test connection to actual GAMA server."""
        # This test would be enabled when testing with real GAMA server
        pass
    
    @pytest.mark.skip(reason="Requires actual GAMA server and model files")
    def test_prison_escape_model_integration(self):
        """Test integration with actual Prison Escape GAMA model."""
        # This test would load the actual .gaml model and run it
        pass
    
    @pytest.mark.skip(reason="Requires actual GAMA server")
    def test_concurrent_environments_real(self):
        """Test multiple environments connecting to GAMA server simultaneously."""
        # This test would verify that multiple environments can connect to GAMA
        pass