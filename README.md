# GAMA-PettingZoo

[![Python Package](https://img.shields.io/pypi/v/gama-pettingzoo)](https://pypi.org/project/gama-pettingzoo/)
[![License](https://img.shields.io/github/license/gama-platform/gama-pettingzoo)](LICENSE)

**GAMA-PettingZoo** is a generic [PettingZoo](https://pettingzoo.farama.org/) environment that enables the integration of simulations from the [GAMA](https://gama-platform.org/) modeling platform with multi-agent reinforcement learning algorithms.

## 🎯 Purpose

This library allows researchers and developers to easily use GAMA models as multi-agent reinforcement learning environments, leveraging the power of GAMA for agent-based modeling and the Python ecosystem for AI.

## ⚡ Quick Start

### Installation

```bash
pip install gama-pettingzoo
```

### Prerequisites

- **GAMA Platform**: Install GAMA from [gama-platform.org](https://gama-platform.org/download)
- **Python 3.8+** with required dependencies

```bash
pip install pettingzoo gama-gymnasium numpy
```

### Basic Usage

```python
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

# Create the environment
env = GamaParallelEnv(
    gaml_experiment_path='your_model.gaml',
    gaml_experiment_name='main',
    gama_ip_address='localhost',
    gama_port=1001
)

# Use as a standard PettingZoo environment
observations, infos = env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation, agent)  # Your policy
    env.step(action)
```

### GamaMultiAgent

The `GamaMultiAgent` is a GAMA agent required in the model to enable interaction between the simulation's learning agents and the PettingZoo environment. It has specialized variables and actions for multi-agent management.

Structure of the agent:

```gaml
species GamaMultiAgent skills:[PettingZooLink] {
    map<string, unknown> action_spaces;
    map<string, unknown> observation_spaces;
    list<string> agents_list;

    map<string, unknown> states;
    map<string, float> rewards;
    map<string, bool> terminated;
    map<string, bool> truncated;
    map<string, unknown> infos;

    map<string, unknown> next_actions;
    map<string, unknown> data;

    action update_data {
        data <- [
            "States"::states,
            "Rewards"::rewards,
            "Terminated"::terminated,
            "Truncated"::truncated,
            "Infos"::infos,
            "Agents"::agents_list
        ];
    }
}
```

### GAMA Configuration

1. **Add the GAMA component** to your model:
   Make sure you have added the species `GamaMultiAgent` described above to your model:

   ```gaml
   species GamaMultiAgent skills:[PettingZooLink];
   ```

   Set up the `action_spaces` and `observation_spaces`:

   ```gaml
   global {
       init {
           create GamaMultiAgent {
               agents_list <- ["prisoner", "guard"];
               action_spaces <- [
                   "prisoner"::["type"::"Discrete", "n"::4],
                   "guard"::["type"::"Discrete", "n"::4]
               ];
               observation_spaces <- [
                   "prisoner"::["type"::"Box", "low"::0, "high"::grid_size, "shape"::[2], "dtype"::"int"],
                   "guard"::["type"::"Box", "low"::0, "high"::grid_size, "shape"::[2], "dtype"::"int"]
               ];
           }
       }
   }
   ```

   Update the multi-agent's data after actions are completed:

   ```gaml
   ask GamaMultiAgent[0] {
       do update_data;
   }
   ```

2. **Launch GAMA in server mode**:

```bash
# Linux/MacOS
./gama-headless.sh -socket 1001

# Windows
gama-headless.bat -socket 1001
```

## 📁 Project Structure

```text
gama-pettingzoo/
├── 📁 src/                    # Main Python package source code
├── 📁 tests/                  # Comprehensive test suite
├── 📁 examples/               # Complete examples and tutorials
│   ├── 📁 Moving Exemple/     # Basic movement example
│   ├── 📁 Pac Man/           # Multi-agent Pac-Man game
│   └── 📁 Prison Escape/     # Prison escape environment
├── 📁 improved_trained_models/ # Advanced trained models
├── 📁 simple_trained_models/   # Basic trained models
├── pyproject.toml             # Python package configuration
├── pytest.ini                # Testing configuration
├── LICENSE                    # Package license
└── README.md                  # This documentation
```

## 📚 Documentation and Examples

### 🚀 Tutorials and Examples

| Example                | Description                                        | Documentation                              |
| ---------------------- | -------------------------------------------------- | ------------------------------------------ |
| **Moving Exemple**     | Introduction to mobile agents                      | [📖 README](examples/Moving%20Exemple/README.md) |
| **Pac Man**           | Multi-agent implementation of Pac-Man game         | [📖 README](examples/Pac%20Man/README.md)        |
| **Prison Escape**     | Prison escape environment (guard vs prisoner)     | [📖 README](examples/Prison%20Escape/README.md)  |

### 📖 Detailed Guides

- **[Moving Exemple Guide](examples/Moving%20Exemple/README.md)**: Complete tutorial for creating your first multi-agent environment
- **[Prison Escape Guide](examples/Prison%20Escape/README.md)**: Escape environment with antagonist agents
- **[Source Code Documentation](src/README.md)**: Technical documentation of the package structure
- **[Testing Guide](tests/README.md)**: Comprehensive testing framework and best practices

## 🛠 Advanced Installation

### From Source Code

```bash
git clone https://github.com/gama-platform/gama-pettingzoo.git
cd gama-pettingzoo
pip install -e src/
```

### Development Dependencies

```bash
pip install -e ".[dev]"      # For development
pip install -e ".[examples]"  # For examples
```

## 🧪 Testing and Validation

```bash
# Run tests
pytest

# With coverage
pytest --cov=gama_pettingzoo --cov-report=html

# Multi-agent specific tests
pytest -m multiagent
```

## 🤖 Supported Algorithms

GAMA-PettingZoo supports all multi-agent reinforcement learning algorithms compatible with PettingZoo:

- **Multi-Agent Q-Learning**
- **Independent Deep Q-Networks (DQN)**
- **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**
- **Multi-Agent Proximal Policy Optimization (PPO)**
- **And many more...**

## 🎮 Available Environments

### Prison Escape
An escape environment where a prisoner attempts to escape while a guard tries to stop them.

- **Agents**: `prisoner`, `guard`
- **Action Space**: Discrete (4 directions)
- **Observation Space**: Agent positions on the grid

### Pac Man Multi-Agent
Multi-agent version of the famous Pac-Man game.

- **Agents**: `pacman`, `ghost1`, `ghost2`, ...
- **Objective**: Cooperation/competition in collecting points

### Moving Exemple
Simple environment for learning the basics of mobile agents.

- **Agents**: Configurable
- **Objective**: Navigation and coordination

## 🤝 Contributing

Contributions are welcome! Check the [issues](https://github.com/gama-platform/gama-pettingzoo/issues) to see how you can help.

### Development

```bash
# Clone the repository
git clone https://github.com/gama-platform/gama-pettingzoo.git
cd gama-pettingzoo

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/ examples/ tests/
isort src/ examples/ tests/
```

## 🔗 Useful Links

- [GAMA Platform](https://gama-platform.org/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [GAMA-Gymnasium](https://github.com/gama-platform/gama-gymnasium)
- [GAMA-Client PyPI](https://pypi.org/project/gama-client/)

---

For more technical details and practical examples, check the documentation in the [`examples/`](examples/) and [`src/`](src/) folders, or explore our comprehensive [testing framework](tests/README.md).