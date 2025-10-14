# GAMA-PettingZoo Examples

This directory contains practical examples of using GAMA-PettingZoo for different multi-agent environments.

## 📁 Available Examples

### 🏃 [Moving Exemple](Moving%20Exemple/)
**Basic environment for learning mobile agents**

- **Difficulty**: Beginner
- **Agents**: Configurable
- **Objective**: Agent navigation and coordination
- **Algorithms**: Q-Learning, DQN
- **Training Time**: ~30 minutes

### 🎮 [Pac Man](Pac%20Man/)
**Multi-agent version of the famous Pac-Man game**

- **Difficulty**: Intermediate
- **Agents**: Pac-Man, Ghosts (multiple)
- **Objective**: Cooperation/competition in a maze
- **Algorithms**: Multi-agent PPO, MADDPG
- **Training Time**: ~2-4 hours

### 🔒 [Prison Escape](Prison%20Escape/)
**Antagonistic escape environment**

- **Difficulty**: Advanced
- **Agents**: Prisoner, Guard
- **Objective**: Escape vs Capture
- **Algorithms**: PPO, Enhanced Q-Learning
- **Training Time**: ~1-3 hours

### 🧪 [Test Examples](test_examples/)
**Testing and validation scripts**

- **Objective**: Environment validation
- **Usage**: Unit and integration tests
- **Audience**: Developers

## 🚀 Getting Started Guide

### Prerequisites for All Examples

1. **GAMA Platform installed** (version 1.8.2+)
2. **Python 3.8+** with `gama-pettingzoo`
3. **Environment configured**:
   ```bash
   pip install gama-pettingzoo[examples]
   ```

### Quick Launch

1. **Start GAMA in server mode**:
   ```bash
   # Linux/MacOS
   ./gama-headless.sh -socket 1001
   
   # Windows
   gama-headless.bat -socket 1001
   ```

2. **Choose an example** and follow its specific README

3. **Basic example**:
   ```bash
   cd "Moving Exemple"
   python animal_proc.py
   ```

## 📊 Examples Comparison

| Example | Complexity | Agents | Time | Recommended for |
|---------|------------|--------|------|-----------------|
| Moving Exemple | ⭐ | 1-N | 30min | Discovery |
| Pac Man | ⭐⭐ | 2-5 | 2h | Learning |
| Prison Escape | ⭐⭐⭐ | 2 | 1-3h | Mastery |

## 🎯 By Learning Objective

### Learn the Basics
👉 Start with **Moving Exemple**
- Fundamental concepts
- PettingZoo API
- GAMA integration

### Multi-Agent Algorithms
👉 Explore **Pac Man**
- Agent cooperation
- Advanced algorithms (MADDPG, PPO)
- Complex environments

### Antagonistic Scenarios
👉 Master **Prison Escape**
- Agent competition
- Adaptive strategies
- Anti-static mechanisms

## 🔧 Common Structure

Each example typically contains:

```text
Example/
├── *.gaml                 # GAMA models
├── controler.gaml         # PettingZoo controller
├── *.py                   # Python training/test scripts
├── requirements.txt       # Specific dependencies
├── README.md             # Example documentation
├── config.py             # Configuration (optional)
└── trained_models/       # Saved models (generated)
```

## 🛠 Example Development

### Creating a New Example

1. **Create folder** with standard structure
2. **Develop GAMA model** (.gaml)
3. **Implement PettingZoo controller**
4. **Write Python training scripts**
5. **Document** with detailed README
6. **Test** complete integration

### Best Practices

- **Clear naming** of files and variables
- **Centralized configuration** in config.py
- **Robust error handling**
- **Parameter documentation**
- **Usage examples** in README

## 🧪 Testing and Validation

Each example should include:
- **Basic tests** (connection, reset, step)
- **Integration tests** with GAMA
- **Validation** of action/observation spaces
- **Complete training examples**

## 🤝 Contributing

To add new examples:

1. **Fork** the repository
2. **Create** your example in this folder
3. **Follow** structure and conventions
4. **Test** thoroughly
5. **Document** in detail
6. **Submit** a Pull Request

## 📚 Useful Resources

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [GAMA Platform](https://gama-platform.org/)
- [GAMA-Gymnasium](https://github.com/gama-platform/gama-gymnasium)
- [Multi-Agent RL Algorithms](https://spinningup.openai.com/en/latest/)

---

💡 **Tip**: Always start with the simplest example and progress towards complexity!