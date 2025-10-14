"""
Configuration file for pytest - GAMA-PettingZoo.

This file contains global pytest configurations, fixtures, and utilities
for the gama-pettingzoo test suite.
"""

import pytest
import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

# Add the source directories to the Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Global pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "multiagent: mark test as multi-agent specific"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gama: mark test as requiring GAMA server"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


@pytest.fixture(scope="session")
def project_paths():
    """Provide commonly used project paths."""
    root = Path(__file__).parent.parent
    return {
        "root": root,
        "src": root / "src",
        "tests": root / "tests",
        "examples": root / "examples",
        "fixtures": root / "tests" / "fixtures",
    }


@pytest.fixture
def sample_multiagent_spaces():
    """Provide sample multi-agent space definitions for testing."""
    return {
        "two_agent_discrete": {
            "prisoner": {"type": "Discrete", "n": 4},
            "guard": {"type": "Discrete", "n": 4}
        },
        "three_agent_mixed": {
            "agent1": {"type": "Discrete", "n": 5},
            "agent2": {"type": "Box", "low": -1.0, "high": 1.0, "shape": [2], "dtype": "float"},
            "agent3": {"type": "MultiBinary", "n": 3}
        },
        "symmetric_agents": {
            f"agent_{i}": {"type": "Box", "low": 0, "high": 10, "shape": [3], "dtype": "int"}
            for i in range(4)
        }
    }


@pytest.fixture
def sample_multiagent_observations():
    """Provide sample multi-agent observations for testing."""
    return {
        "two_agent_simple": {
            "prisoner": np.array([0, 1]),
            "guard": np.array([6, 5])
        },
        "three_agent_mixed": {
            "agent1": 2,
            "agent2": np.array([0.5, -0.3]),
            "agent3": np.array([1, 0, 1])
        },
        "prison_escape_example": {
            "prisoner": np.array([0, 0, 6, 6, 6, 6]),  # [prisoner_x, prisoner_y, guard_x, guard_y, exit_x, exit_y]
            "guard": np.array([3, 3, 0, 0, 6, 6])     # [guard_x, guard_y, prisoner_x, prisoner_y, exit_x, exit_y]
        }
    }


@pytest.fixture
def sample_multiagent_actions():
    """Provide sample multi-agent actions for testing."""
    return {
        "two_agent_discrete": {
            "prisoner": 1,  # Move right
            "guard": 3      # Move down
        },
        "three_agent_mixed": {
            "agent1": 2,
            "agent2": np.array([0.8, -0.2]),
            "agent3": np.array([0, 1, 1])
        },
        "prison_escape_actions": {
            "prisoner": 0,  # Stay
            "guard": 2      # Move up
        }
    }


@pytest.fixture
def sample_multiagent_rewards():
    """Provide sample multi-agent rewards for testing."""
    return {
        "balanced_rewards": {
            "prisoner": 1.0,
            "guard": -1.0
        },
        "all_positive": {
            "agent1": 0.5,
            "agent2": 1.0,
            "agent3": 0.3
        },
        "mixed_rewards": {
            "cooperator": 2.0,
            "defector": -0.5,
            "neutral": 0.0
        }
    }


@pytest.fixture
def sample_multiagent_info():
    """Provide sample multi-agent info dictionaries for testing."""
    return {
        "basic_info": {
            "prisoner": {"position": [0, 1], "health": 100},
            "guard": {"position": [3, 3], "stamina": 95}
        },
        "prison_escape_info": {
            "prisoner": {
                "position": [0, 1],
                "moves_made": 1,
                "distance_to_exit": 7.07
            },
            "guard": {
                "position": [3, 2], 
                "moves_made": 2,
                "distance_to_prisoner": 3.16
            }
        }
    }


@pytest.fixture
def sample_termination_truncation():
    """Provide sample termination and truncation states."""
    return {
        "prisoner_wins": {
            "terminations": {"prisoner": True, "guard": True},
            "truncations": {"prisoner": False, "guard": False}
        },
        "guard_wins": {
            "terminations": {"prisoner": True, "guard": True},
            "truncations": {"prisoner": False, "guard": False}
        },
        "time_limit": {
            "terminations": {"prisoner": False, "guard": False},
            "truncations": {"prisoner": True, "guard": True}
        },
        "ongoing": {
            "terminations": {"prisoner": False, "guard": False},
            "truncations": {"prisoner": False, "guard": False}
        }
    }


@pytest.fixture
def sample_gama_multiagent_data():
    """Provide sample GAMA multi-agent step data."""
    return {
        "prison_escape_step": {
            "States": {
                "prisoner": [0, 1, 3, 3, 6, 6],
                "guard": [3, 3, 0, 1, 6, 6]
            },
            "Rewards": {
                "prisoner": 0.0,
                "guard": 0.0
            },
            "Terminated": {
                "prisoner": False,
                "guard": False
            },
            "Truncated": {
                "prisoner": False,
                "guard": False
            },
            "Infos": {
                "prisoner": {"step": 1, "distance_to_exit": 7.07},
                "guard": {"step": 1, "distance_to_prisoner": 3.16}
            },
            "Agents": ["prisoner", "guard"]
        },
        "terminal_step": {
            "States": {
                "prisoner": [6, 6, 3, 3, 6, 6],
                "guard": [3, 3, 6, 6, 6, 6]
            },
            "Rewards": {
                "prisoner": 1.0,
                "guard": -1.0
            },
            "Terminated": {
                "prisoner": True,
                "guard": True
            },
            "Truncated": {
                "prisoner": False,
                "guard": False
            },
            "Infos": {
                "prisoner": {"step": 50, "reason": "escaped"},
                "guard": {"step": 50, "reason": "prisoner_escaped"}
            },
            "Agents": ["prisoner", "guard"]
        }
    }


@pytest.fixture
def mock_gama_parallel_env():
    """Provide a mock GamaParallelEnv for testing."""
    from unittest.mock import MagicMock
    
    env = MagicMock()
    
    # Set up default properties
    env.agents = ["prisoner", "guard"]
    env.num_agents = 2
    env.possible_agents = ["prisoner", "guard"]
    
    # Set up default method returns
    env.reset.return_value = (
        {"prisoner": np.array([0, 0, 3, 3, 6, 6]), "guard": np.array([3, 3, 0, 0, 6, 6])},
        {"prisoner": {}, "guard": {}}
    )
    
    env.step.return_value = (
        {"prisoner": np.array([0, 1, 3, 3, 6, 6]), "guard": np.array([3, 3, 0, 1, 6, 6])},
        {"prisoner": 0.0, "guard": 0.0},
        {"prisoner": False, "guard": False},
        {"prisoner": False, "guard": False},
        {"prisoner": {"step": 1}, "guard": {"step": 1}}
    )
    
    return env


@pytest.fixture
def mock_gama_aec_env():
    """Provide a mock GamaAECEnv for testing."""
    from unittest.mock import MagicMock
    
    env = MagicMock()
    
    # Set up AEC-specific properties
    env.agents = ["prisoner", "guard"]
    env.num_agents = 2
    env.possible_agents = ["prisoner", "guard"]
    env.agent_selection = "prisoner"
    
    # Set up AEC methods
    env.reset.return_value = None
    env.step.return_value = None
    env.observe.return_value = np.array([0, 0, 3, 3, 6, 6])
    env.last.return_value = (
        np.array([0, 0, 3, 3, 6, 6]),  # observation
        0.0,  # reward
        False,  # termination
        False,  # truncation
        {}  # info
    )
    
    return env


@pytest.fixture
def sample_agent_configs():
    """Provide sample agent configurations for testing."""
    return {
        "prison_escape": {
            "agents_list": ["prisoner", "guard"],
            "action_spaces": {
                "prisoner": {"type": "Discrete", "n": 4},
                "guard": {"type": "Discrete", "n": 4}
            },
            "observation_spaces": {
                "prisoner": {"type": "Box", "low": 0, "high": 6, "shape": [6], "dtype": "int"},
                "guard": {"type": "Box", "low": 0, "high": 6, "shape": [6], "dtype": "int"}
            }
        },
        "moving_example": {
            "agents_list": ["agent1", "agent2"],
            "action_spaces": {
                "agent1": {"type": "Discrete", "n": 5},
                "agent2": {"type": "Discrete", "n": 5}
            },
            "observation_spaces": {
                "agent1": {"type": "Box", "low": 0, "high": 20, "shape": [4], "dtype": "float"},
                "agent2": {"type": "Box", "low": 0, "high": 20, "shape": [4], "dtype": "float"}
            }
        }
    }


# Test utilities
def assert_multiagent_observations_valid(observations: Dict[str, Any], agents: List[str]):
    """Utility function to validate multi-agent observations."""
    assert isinstance(observations, dict), "Observations must be a dictionary"
    assert len(observations) == len(agents), f"Expected {len(agents)} observations, got {len(observations)}"
    
    for agent in agents:
        assert agent in observations, f"Missing observation for agent {agent}"
        assert observations[agent] is not None, f"Observation for {agent} is None"


def assert_multiagent_actions_valid(actions: Dict[str, Any], agents: List[str]):
    """Utility function to validate multi-agent actions."""
    assert isinstance(actions, dict), "Actions must be a dictionary"
    assert len(actions) == len(agents), f"Expected {len(agents)} actions, got {len(actions)}"
    
    for agent in agents:
        assert agent in actions, f"Missing action for agent {agent}"
        assert actions[agent] is not None, f"Action for {agent} is None"


def assert_multiagent_rewards_valid(rewards: Dict[str, float], agents: List[str]):
    """Utility function to validate multi-agent rewards."""
    assert isinstance(rewards, dict), "Rewards must be a dictionary"
    assert len(rewards) == len(agents), f"Expected {len(agents)} rewards, got {len(rewards)}"
    
    for agent in agents:
        assert agent in rewards, f"Missing reward for agent {agent}"
        assert isinstance(rewards[agent], (int, float, np.number)), f"Reward for {agent} must be numeric"


def assert_multiagent_dones_valid(terminations: Dict[str, bool], truncations: Dict[str, bool], agents: List[str]):
    """Utility function to validate multi-agent done states."""
    assert isinstance(terminations, dict), "Terminations must be a dictionary"
    assert isinstance(truncations, dict), "Truncations must be a dictionary"
    
    for agent in agents:
        assert agent in terminations, f"Missing termination for agent {agent}"
        assert agent in truncations, f"Missing truncation for agent {agent}"
        assert isinstance(terminations[agent], bool), f"Termination for {agent} must be boolean"
        assert isinstance(truncations[agent], bool), f"Truncation for {agent} must be boolean"


def create_mock_gama_multiagent_response(success=True, data=None):
    """Create a mock GAMA multi-agent response."""
    if data is None:
        data = {
            "States": {"agent1": [0, 1], "agent2": [2, 3]},
            "Rewards": {"agent1": 0.0, "agent2": 0.0},
            "Terminated": {"agent1": False, "agent2": False},
            "Truncated": {"agent1": False, "agent2": False},
            "Infos": {"agent1": {}, "agent2": {}},
            "Agents": ["agent1", "agent2"]
        }
    
    return {
        "type": "CommandExecutedSuccessfully" if success else "Error",
        "content": str(data) if success else "Error occurred"
    }


@pytest.fixture
def performance_benchmark():
    """Provide performance benchmarking utilities."""
    import time
    from contextlib import contextmanager
    
    class PerformanceBenchmark:
        def __init__(self):
            self.results = {}
        
        @contextmanager
        def time_operation(self, operation_name: str):
            start_time = time.time()
            try:
                yield
            finally:
                end_time = time.time()
                self.results[operation_name] = end_time - start_time
        
        def get_result(self, operation_name: str) -> float:
            return self.results.get(operation_name, 0.0)
        
        def assert_performance(self, operation_name: str, max_time: float):
            actual_time = self.get_result(operation_name)
            assert actual_time <= max_time, (
                f"Operation {operation_name} took {actual_time:.3f}s, "
                f"expected <= {max_time:.3f}s"
            )
    
    return PerformanceBenchmark()


# Parameterized test data for multi-agent scenarios
@pytest.fixture(params=[2, 3, 4])
def num_agents(request):
    """Parameterized fixture for different numbers of agents."""
    return request.param


@pytest.fixture(params=[
    {"type": "Discrete", "n": 4},
    {"type": "Box", "low": -1, "high": 1, "shape": [2], "dtype": "float"},
    {"type": "MultiBinary", "n": 3}
])
def sample_action_space(request):
    """Parameterized fixture for different action space types."""
    return request.param