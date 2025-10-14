# GAMA-PettingZoo Source Code Documentation

This folder contains the main source code of the GAMA-PettingZoo package.

## 📁 Structure

```text
src/
└── gama_pettingzoo/           # Main package
    ├── __init__.py            # Package initialization
    ├── gama_parallel_env.py   # PettingZoo parallel environment
    ├── gama_aec_env.py        # AEC (Agent Environment Cycle) environment
    └── utils/                 # Utilities and helpers
        ├── __init__.py
        ├── connection.py      # GAMA connection management
        ├── spaces.py         # Action/observation space definitions
        └── converters.py     # GAMA/PettingZoo data converters
```

## 🔧 Main Components

### `gama_parallel_env.py`
Implementation of PettingZoo parallel environment that enables:
- Simultaneous execution of all agents
- Communication with GAMA via sockets
- Multi-agent action and observation space management
- Simulation cycle synchronization

### `gama_aec_env.py`
AEC (Agent Environment Cycle) environment for:
- Sequential agent execution
- Fine-grained execution order control
- Compatibility with algorithms requiring turn-based execution

### `utils/`
Utility modules for:
- **`connection.py`** : TCP/IP communication management with GAMA
- **`spaces.py`** : GAMA to PettingZoo space conversion
- **`converters.py`** : Data transformation between formats

## 🚀 Main API

### GamaParallelEnv

```python
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

env = GamaParallelEnv(
    gaml_experiment_path='model.gaml',
    gaml_experiment_name='main',
    gama_ip_address='localhost',
    gama_port=1001
)
```

### GamaAECEnv

```python
from gama_pettingzoo.gama_aec_env import GamaAECEnv

env = GamaAECEnv(
    gaml_experiment_path='model.gaml',
    gaml_experiment_name='main',
    gama_ip_address='localhost',
    gama_port=1001
)
```

## 🔄 Lifecycle

1. **Initialization** : Connect to GAMA and retrieve spaces
2. **Reset** : Initialize simulation and agents
3. **Step** : Execute actions and retrieve observations
4. **Close** : Clean GAMA connection closure

## 🧪 Tests

Corresponding tests are located in the `tests/` folder at the project root.

## 📖 Developer Documentation

To contribute to development:

1. **Code structure** : Follow PEP 8 and Python conventions
2. **Tests** : Add tests for any new functionality  
3. **Documentation** : Documenter public APIs with docstrings
4. **Type hints** : Use type annotations for better readability

## 🔗 Dependencies

- `pettingzoo>=1.22.0` : Multi-agent framework
- `gama_gymnasium>=0.1.0` : GAMA integration for Gymnasium
- `numpy>=1.21.0` : Numerical computations
- `typing` : Type annotations (Python < 3.9)