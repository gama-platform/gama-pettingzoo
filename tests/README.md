# GAMA-PettingZoo Tests

This directory contains the comprehensive test suite for GAMA-PettingZoo, including unit tests, integration tests, and performance tests.

## 📁 Test Structure

```text
tests/
├── unit/                  # Unit tests
│   ├── test_env.py       # Environment tests
│   ├── test_spaces.py    # Action/observation space tests
│   └── test_utils.py     # Utility tests
├── integration/          # Integration tests
│   ├── test_gama_connection.py  # GAMA connection tests
│   └── test_full_workflow.py   # Complete workflow tests
├── fixtures/             # Test data and configurations
│   ├── test_models/      # GAMA models for testing
│   └── configs/          # Test configurations
└── README.md            # This documentation
```

## 🚀 Running Tests

### Complete Test Suite
```bash
# All tests
pytest

# With code coverage
pytest --cov=gama_pettingzoo --cov-report=html --cov-report=xml
```

### Tests by Category
```bash
# Unit tests only
pytest -m unit

# Integration tests (require GAMA)
pytest -m integration

# Multi-agent specific tests
pytest -m multiagent

# Exclude slow tests (default)
pytest -m "not slow"
```

### Performance Tests
```bash
# Performance tests
pytest -m performance --durations=10

# Benchmark tests
pytest --benchmark-only
```

## 🧪 Test Types

### Unit Tests (`unit/`)
- **Isolation**: Testing individual components
- **Speed**: Execution < 1 second each
- **Mocking**: Using mocks for GAMA
- **Coverage**: Targets 90%+ of code

### Integration Tests (`integration/`)
- **GAMA Required**: Tests with real GAMA instance
- **Complete Workflow**: From connection to closure
- **Real Environments**: Tests with .gaml models
- **Performance**: Latency and throughput metrics

### Test Fixtures (`fixtures/`)
- **Test Models**: Simplified GAMA environments
- **Configurations**: Standardized parameters
- **Sample Data**: Typical observations and actions

## ⚙️ Test Configuration

### Pytest Markers

Defined in `pytest.ini`:
- `unit`: Fast unit tests
- `integration`: Tests requiring GAMA
- `multiagent`: Multi-agent specific tests
- `performance`: Performance tests
- `slow`: Long tests (> 30 seconds)
- `gama`: Tests requiring GAMA platform

### Environment Variables

```bash
# GAMA configuration for tests
export GAMA_TEST_HOST=localhost
export GAMA_TEST_PORT=1001
export GAMA_TEST_TIMEOUT=30

# Test configuration
export PYTEST_TIMEOUT=60
export PYTEST_WORKERS=auto
```

## 🔧 Écriture de nouveaux tests

### Structure type d'un test

```python
import pytest
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

class TestGamaParallelEnv:
    """Tests pour l'environnement parallèle."""
    
    @pytest.fixture
    def simple_env(self):
        """Fixture d'environnement simple."""
        return GamaParallelEnv(
            gaml_experiment_path='fixtures/test_models/simple.gaml',
            gaml_experiment_name='test',
            gama_ip_address='localhost',
            gama_port=1001
        )
    
    @pytest.mark.unit
    def test_initialization(self, simple_env):
        """Test d'initialisation basique."""
        assert simple_env is not None
        assert simple_env.num_agents > 0
    
    @pytest.mark.integration
    @pytest.mark.gama
    def test_reset_and_step(self, simple_env):
        """Test du cycle reset/step."""
        obs, info = simple_env.reset()
        assert obs is not None
        
        actions = {agent: 0 for agent in simple_env.agents}
        obs, rewards, terms, truncs, infos = simple_env.step(actions)
        assert len(obs) == len(simple_env.agents)
```

### Bonnes pratiques

1. **Nommage** : `test_<fonctionnalité>_<scénario>`
2. **Isolation** : Chaque test indépendant
3. **Fixtures** : Réutiliser les configurations communes
4. **Assertions** : Claires et spécifiques
5. **Documentation** : Docstrings explicatives

## 📊 Code Coverage

### Coverage Targets
- **Overall**: > 85%
- **Core Modules**: > 90%
- **Utilities**: > 95%
- **Examples**: > 70%

### Coverage Reports
```bash
# Generate HTML report
pytest --cov=gama_pettingzoo --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🚀 Automated Testing (CI/CD)

### GitHub Actions

Configuration in `.github/workflows/tests.yml`:
- **Unit Tests**: On every commit
- **Integration Tests**: On main PRs
- **Performance Tests**: Weekly
- **Multiple Python Versions**: 3.8, 3.9, 3.10, 3.11

### Pre-commit Hooks

```bash
# Install hooks
pip install pre-commit
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

## 🐛 Dépannage des tests

### Problèmes fréquents

1. **"Tests d'intégration échouent"**
   ```bash
   # Vérifier GAMA
   ./gama-headless.sh -socket 1001 &
   pytest -m integration -v
   ```

2. **"Timeout des tests"**
   ```bash
   # Augmenter le timeout
   pytest --timeout=120
   ```

3. **"Fixtures non trouvées"**
   ```bash
   # Vérifier le PYTHONPATH
   export PYTHONPATH=$PWD/tests:$PYTHONPATH
   ```

### Débogage avancé

```bash
# Mode verbose maximum
pytest -vvv --tb=long

# Debug avec pdb
pytest --pdb

# Profile des tests lents
pytest --durations=0
```

## 📈 Métriques de qualité

### Objectifs
- **Temps d'exécution** : < 5 minutes (tests complets)
- **Fiabilité** : > 99% de succès
- **Maintenance** : Tests maintenus à jour avec le code
- **Documentation** : 100% des tests documentés

### Surveillance continue
- **Performance** : Détection des régressions
- **Flakiness** : Identification des tests instables
- **Couverture** : Suivi de l'évolution

## 🤝 Contribution aux tests

### Ajout de nouveaux tests
1. **Identifier** la fonctionnalité à tester
2. **Choisir** le type approprié (unit/integration)
3. **Écrire** le test avec la structure standard
4. **Vérifier** la couverture ajoutée
5. **Documenter** le comportement testé

### Review des tests
- **Lisibilité** : Tests faciles à comprendre
- **Pertinence** : Tests de cas réalistes
- **Performance** : Tests rapides et efficaces
- **Maintenance** : Tests faciles à maintenir

---

🧪 **Tip**: Write tests alongside the code, not after!