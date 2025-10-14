"""
Tests for fixtures and test utilities.

This module tests the test fixtures and utilities defined in conftest.py
to ensure they work correctly.
"""

import pytest
import numpy as np
from pathlib import Path


class TestFixtures:
    """Test the fixtures defined in conftest.py."""
    
    def test_project_paths_fixture(self, project_paths):
        """Test that project_paths fixture provides correct paths."""
        assert "root" in project_paths
        assert "src" in project_paths
        assert "tests" in project_paths
        assert "examples" in project_paths
        assert "fixtures" in project_paths
        
        # All paths should be Path objects
        for path_name, path_obj in project_paths.items():
            assert isinstance(path_obj, Path), f"{path_name} should be a Path object"
            
        # Verify logical relationships
        assert project_paths["src"].parent == project_paths["root"]
        assert project_paths["tests"].parent == project_paths["root"]
        assert project_paths["examples"].parent == project_paths["root"]
        assert project_paths["fixtures"].parent == project_paths["tests"]
    
    def test_sample_multiagent_spaces(self, sample_multiagent_spaces):
        """Test sample multi-agent spaces fixture."""
        assert "two_agent_discrete" in sample_multiagent_spaces
        assert "three_agent_mixed" in sample_multiagent_spaces
        assert "symmetric_agents" in sample_multiagent_spaces
        
        # Check two_agent_discrete structure
        two_agent = sample_multiagent_spaces["two_agent_discrete"]
        assert "prisoner" in two_agent
        assert "guard" in two_agent
        assert two_agent["prisoner"]["type"] == "Discrete"
        assert two_agent["guard"]["type"] == "Discrete"
        
        # Check three_agent_mixed has different types
        three_agent = sample_multiagent_spaces["three_agent_mixed"]
        space_types = {agent_data["type"] for agent_data in three_agent.values()}
        assert len(space_types) > 1, "Should have mixed space types"
        
        # Check symmetric_agents has 4 agents
        symmetric = sample_multiagent_spaces["symmetric_agents"]
        assert len(symmetric) == 4
        
        # All agents in symmetric should have same space type
        symmetric_types = {agent_data["type"] for agent_data in symmetric.values()}
        assert len(symmetric_types) == 1, "Symmetric agents should have same space type"
    
    def test_sample_multiagent_observations(self, sample_multiagent_observations):
        """Test sample multi-agent observations fixture."""
        assert "two_agent_simple" in sample_multiagent_observations
        assert "three_agent_mixed" in sample_multiagent_observations
        assert "prison_escape_example" in sample_multiagent_observations
        
        # Check two_agent_simple
        two_agent_obs = sample_multiagent_observations["two_agent_simple"]
        assert "prisoner" in two_agent_obs
        assert "guard" in two_agent_obs
        assert isinstance(two_agent_obs["prisoner"], np.ndarray)
        assert isinstance(two_agent_obs["guard"], np.ndarray)
        
        # Check three_agent_mixed has different observation types
        three_agent_obs = sample_multiagent_observations["three_agent_mixed"]
        assert len(three_agent_obs) == 3
        
        # agent1 should be integer, agent2 should be array, agent3 should be array
        assert isinstance(three_agent_obs["agent1"], (int, np.integer))
        assert isinstance(three_agent_obs["agent2"], np.ndarray)
        assert isinstance(three_agent_obs["agent3"], np.ndarray)
        
        # Prison escape should have proper structure
        prison_obs = sample_multiagent_observations["prison_escape_example"]
        assert len(prison_obs["prisoner"]) == 6  # [prisoner_x, prisoner_y, guard_x, guard_y, exit_x, exit_y]
        assert len(prison_obs["guard"]) == 6     # [guard_x, guard_y, prisoner_x, prisoner_y, exit_x, exit_y]
    
    def test_sample_multiagent_actions(self, sample_multiagent_actions):
        """Test sample multi-agent actions fixture."""
        assert "two_agent_discrete" in sample_multiagent_actions
        assert "three_agent_mixed" in sample_multiagent_actions
        assert "prison_escape_actions" in sample_multiagent_actions
        
        # Check action types are appropriate
        two_agent_actions = sample_multiagent_actions["two_agent_discrete"]
        assert isinstance(two_agent_actions["prisoner"], (int, np.integer))
        assert isinstance(two_agent_actions["guard"], (int, np.integer))
        
        # Check mixed actions
        mixed_actions = sample_multiagent_actions["three_agent_mixed"]
        assert isinstance(mixed_actions["agent1"], (int, np.integer))
        assert isinstance(mixed_actions["agent2"], np.ndarray)
        assert isinstance(mixed_actions["agent3"], np.ndarray)
    
    def test_sample_multiagent_rewards(self, sample_multiagent_rewards):
        """Test sample multi-agent rewards fixture."""
        assert "balanced_rewards" in sample_multiagent_rewards
        assert "all_positive" in sample_multiagent_rewards
        assert "mixed_rewards" in sample_multiagent_rewards
        
        # Check balanced rewards sum to zero
        balanced = sample_multiagent_rewards["balanced_rewards"]
        total_reward = sum(balanced.values())
        assert abs(total_reward) < 1e-10, "Balanced rewards should sum to approximately zero"
        
        # Check all positive rewards
        all_positive = sample_multiagent_rewards["all_positive"]
        assert all(reward > 0 for reward in all_positive.values()), "All rewards should be positive"
        
        # Check mixed rewards have both positive and negative
        mixed = sample_multiagent_rewards["mixed_rewards"]
        rewards_values = list(mixed.values())
        has_positive = any(reward > 0 for reward in rewards_values)
        has_negative = any(reward < 0 for reward in rewards_values)
        has_zero = any(reward == 0 for reward in rewards_values)
        assert has_positive and (has_negative or has_zero), "Mixed rewards should have variety"
    
    def test_sample_multiagent_info(self, sample_multiagent_info):
        """Test sample multi-agent info fixture."""
        assert "basic_info" in sample_multiagent_info
        assert "prison_escape_info" in sample_multiagent_info
        
        # Check basic info structure
        basic = sample_multiagent_info["basic_info"]
        assert "prisoner" in basic and "guard" in basic
        assert "position" in basic["prisoner"]
        assert "health" in basic["prisoner"]
        
        # Check prison escape info has expected fields
        prison_info = sample_multiagent_info["prison_escape_info"]
        assert "position" in prison_info["prisoner"]
        assert "moves_made" in prison_info["prisoner"]
        assert "distance_to_exit" in prison_info["prisoner"]
        assert "distance_to_prisoner" in prison_info["guard"]
    
    def test_sample_termination_truncation(self, sample_termination_truncation):
        """Test sample termination/truncation states fixture."""
        assert "prisoner_wins" in sample_termination_truncation
        assert "guard_wins" in sample_termination_truncation
        assert "time_limit" in sample_termination_truncation
        assert "ongoing" in sample_termination_truncation
        
        # Check prisoner wins scenario
        prisoner_wins = sample_termination_truncation["prisoner_wins"]
        assert prisoner_wins["terminations"]["prisoner"] is True
        assert prisoner_wins["terminations"]["guard"] is True
        assert prisoner_wins["truncations"]["prisoner"] is False
        assert prisoner_wins["truncations"]["guard"] is False
        
        # Check time limit scenario
        time_limit = sample_termination_truncation["time_limit"]
        assert all(not term for term in time_limit["terminations"].values())
        assert all(trunc for trunc in time_limit["truncations"].values())
        
        # Check ongoing scenario
        ongoing = sample_termination_truncation["ongoing"]
        assert all(not term for term in ongoing["terminations"].values())
        assert all(not trunc for trunc in ongoing["truncations"].values())
    
    def test_sample_gama_multiagent_data(self, sample_gama_multiagent_data):
        """Test sample GAMA multi-agent data fixture."""
        assert "prison_escape_step" in sample_gama_multiagent_data
        assert "terminal_step" in sample_gama_multiagent_data
        
        # Check prison escape step structure
        step_data = sample_gama_multiagent_data["prison_escape_step"]
        required_keys = ["States", "Rewards", "Terminated", "Truncated", "Infos", "Agents"]
        for key in required_keys:
            assert key in step_data, f"Missing key {key} in step data"
        
        # Check agents consistency
        agents = step_data["Agents"]
        for key in ["States", "Rewards", "Terminated", "Truncated", "Infos"]:
            assert set(step_data[key].keys()) == set(agents), f"Agent mismatch in {key}"
        
        # Check terminal step has proper terminations
        terminal_data = sample_gama_multiagent_data["terminal_step"]
        assert all(terminal_data["Terminated"].values()), "Terminal step should have all agents terminated"


class TestTestUtilities:
    """Test the utility functions defined in conftest.py."""
    
    def test_assert_multiagent_observations_valid(self, sample_multiagent_observations):
        """Test the observation validation utility."""
        from conftest import assert_multiagent_observations_valid
        
        observations = sample_multiagent_observations["two_agent_simple"]
        agents = ["prisoner", "guard"]
        
        # Should not raise any exception for valid observations
        assert_multiagent_observations_valid(observations, agents)
        
        # Test with missing agent
        invalid_obs = {"prisoner": np.array([0, 1])}  # Missing guard
        with pytest.raises(AssertionError, match="Expected 2 observations, got 1"):
            assert_multiagent_observations_valid(invalid_obs, agents)
        
        # Test with None observation
        invalid_obs = {"prisoner": None, "guard": np.array([0, 1])}
        with pytest.raises(AssertionError, match="Observation for prisoner is None"):
            assert_multiagent_observations_valid(invalid_obs, agents)
    
    def test_assert_multiagent_actions_valid(self, sample_multiagent_actions):
        """Test the action validation utility."""
        from conftest import assert_multiagent_actions_valid
        
        actions = sample_multiagent_actions["two_agent_discrete"]
        agents = ["prisoner", "guard"]
        
        # Should not raise any exception for valid actions
        assert_multiagent_actions_valid(actions, agents)
        
        # Test with missing agent action
        invalid_actions = {"prisoner": 1}  # Missing guard
        with pytest.raises(AssertionError, match="Expected 2 actions, got 1"):
            assert_multiagent_actions_valid(invalid_actions, agents)
    
    def test_assert_multiagent_rewards_valid(self, sample_multiagent_rewards):
        """Test the reward validation utility."""
        from conftest import assert_multiagent_rewards_valid
        
        rewards = sample_multiagent_rewards["balanced_rewards"]
        agents = ["prisoner", "guard"]
        
        # Should not raise any exception for valid rewards
        assert_multiagent_rewards_valid(rewards, agents)
        
        # Test with invalid reward type
        invalid_rewards = {"prisoner": "not_a_number", "guard": 0.5}
        with pytest.raises(AssertionError, match="Reward for prisoner must be numeric"):
            assert_multiagent_rewards_valid(invalid_rewards, agents)
    
    def test_assert_multiagent_dones_valid(self, sample_termination_truncation):
        """Test the done states validation utility."""
        from conftest import assert_multiagent_dones_valid
        
        done_data = sample_termination_truncation["ongoing"]
        terminations = done_data["terminations"]
        truncations = done_data["truncations"]
        agents = ["prisoner", "guard"]
        
        # Should not raise any exception for valid done states
        assert_multiagent_dones_valid(terminations, truncations, agents)
        
        # Test with missing termination
        invalid_terminations = {"prisoner": False}  # Missing guard
        with pytest.raises(AssertionError, match="Missing termination for agent guard"):
            assert_multiagent_dones_valid(invalid_terminations, truncations, agents)
    
    def test_create_mock_gama_multiagent_response(self):
        """Test the mock GAMA response creator utility."""
        from conftest import create_mock_gama_multiagent_response
        
        # Test successful response with default data
        success_response = create_mock_gama_multiagent_response(success=True)
        assert success_response["type"] == "CommandExecutedSuccessfully"
        assert "content" in success_response
        
        # Test error response
        error_response = create_mock_gama_multiagent_response(success=False)
        assert error_response["type"] == "Error"
        assert error_response["content"] == "Error occurred"
        
        # Test with custom data
        custom_data = {"States": {"agent1": [1, 2, 3]}, "Agents": ["agent1"]}
        custom_response = create_mock_gama_multiagent_response(success=True, data=custom_data)
        assert str(custom_data) in custom_response["content"]


class TestPerformanceBenchmark:
    """Test the performance benchmark utility."""
    
    def test_performance_benchmark_basic(self, performance_benchmark):
        """Test basic performance benchmark functionality."""
        import time
        
        # Test timing an operation
        with performance_benchmark.time_operation("test_operation"):
            time.sleep(0.01)  # Sleep for 10ms
        
        # Should have recorded the operation
        result = performance_benchmark.get_result("test_operation")
        assert result >= 0.01  # Should be at least 10ms
        assert result < 0.5    # Should be less than 500ms (reasonable upper bound)
    
    def test_performance_benchmark_assert(self, performance_benchmark):
        """Test performance benchmark assertion."""
        import time
        
        # Time a fast operation
        with performance_benchmark.time_operation("fast_operation"):
            time.sleep(0.001)  # Sleep for 1ms
        
        # Should pass assertion for reasonable limit
        performance_benchmark.assert_performance("fast_operation", max_time=0.1)
        
        # Should fail assertion for unreasonable limit
        with pytest.raises(AssertionError, match="took .* expected <="):
            performance_benchmark.assert_performance("fast_operation", max_time=0.0001)
    
    def test_performance_benchmark_multiple_operations(self, performance_benchmark):
        """Test performance benchmark with multiple operations."""
        import time
        
        # Time multiple operations
        operations = ["op1", "op2", "op3"]
        
        for op in operations:
            with performance_benchmark.time_operation(op):
                time.sleep(0.001)
        
        # Should have results for all operations
        for op in operations:
            result = performance_benchmark.get_result(op)
            assert result > 0
        
        # Should return 0 for non-existent operation
        assert performance_benchmark.get_result("nonexistent") == 0.0


class TestParametrizedFixtures:
    """Test parametrized fixtures."""
    
    def test_num_agents_parametrized(self, num_agents):
        """Test the parametrized num_agents fixture."""
        assert isinstance(num_agents, int)
        assert num_agents >= 2
        assert num_agents <= 4
    
    def test_sample_action_space_parametrized(self, sample_action_space):
        """Test the parametrized action space fixture."""
        assert isinstance(sample_action_space, dict)
        assert "type" in sample_action_space
        
        space_type = sample_action_space["type"]
        assert space_type in ["Discrete", "Box", "MultiBinary"]
        
        if space_type == "Discrete":
            assert "n" in sample_action_space
            assert isinstance(sample_action_space["n"], int)
        elif space_type == "Box":
            assert "low" in sample_action_space
            assert "high" in sample_action_space
            assert "shape" in sample_action_space
            assert "dtype" in sample_action_space
        elif space_type == "MultiBinary":
            assert "n" in sample_action_space
            assert isinstance(sample_action_space["n"], int)


class TestMockEnvironments:
    """Test mock environment fixtures."""
    
    def test_mock_gama_parallel_env(self, mock_gama_parallel_env):
        """Test the mock parallel environment fixture."""
        env = mock_gama_parallel_env
        
        # Should have basic properties
        assert hasattr(env, 'agents')
        assert hasattr(env, 'num_agents')
        assert hasattr(env, 'possible_agents')
        
        # Should have methods
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        
        # Methods should return expected structures
        observations, infos = env.reset.return_value
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        
        step_result = env.step.return_value
        assert len(step_result) == 5  # obs, rewards, terminations, truncations, infos
    
    def test_mock_gama_aec_env(self, mock_gama_aec_env):
        """Test the mock AEC environment fixture."""
        env = mock_gama_aec_env
        
        # Should have AEC-specific properties
        assert hasattr(env, 'agent_selection')
        assert hasattr(env, 'agents')
        
        # Should have AEC methods
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'observe')
        assert hasattr(env, 'last')
        
        # Methods should return expected types
        observation = env.observe.return_value
        assert isinstance(observation, np.ndarray)
        
        last_result = env.last.return_value
        assert len(last_result) == 5  # observation, reward, termination, truncation, info


class TestAgentConfigs:
    """Test agent configuration fixtures."""
    
    def test_sample_agent_configs(self, sample_agent_configs):
        """Test the sample agent configurations fixture."""
        assert "prison_escape" in sample_agent_configs
        assert "moving_example" in sample_agent_configs
        
        # Check prison escape config
        prison_config = sample_agent_configs["prison_escape"]
        assert "agents_list" in prison_config
        assert "action_spaces" in prison_config
        assert "observation_spaces" in prison_config
        
        agents = prison_config["agents_list"]
        assert "prisoner" in agents
        assert "guard" in agents
        
        # Action and observation spaces should match agents
        action_spaces = prison_config["action_spaces"]
        obs_spaces = prison_config["observation_spaces"]
        
        for agent in agents:
            assert agent in action_spaces
            assert agent in obs_spaces
            assert "type" in action_spaces[agent]
            assert "type" in obs_spaces[agent]
        
        # Check moving example config
        moving_config = sample_agent_configs["moving_example"]
        assert len(moving_config["agents_list"]) == 2
        assert all("agent" in agent for agent in moving_config["agents_list"])