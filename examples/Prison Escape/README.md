# Prison Escape - Multi-Agent Training Environment# Entraînement d'agents dans Prison Escape avec GAMA



A multi-agent reinforcement learning environment implemented with GAMA-PettingZoo. Train two competing agents (prisoner and guard) in a strategic escape scenario using advanced RL algorithms.Ce projet permet d'entraîner deux agents IA (prisoner et guard) dans l'environnement Prison Escape de GAMA en utilisant l'apprentissage par renforcement avec l'algorithme PPO (Proximal Policy Optimization).



## 🎯 Objective## 🎯 Objectif



- **Prisoner**: Reach the escape cell (brown) without being captured- **Prisoner** : Doit atteindre la case d'évasion (brune) sans se faire capturer

- **Guard**: Capture the prisoner before they reach the exit- **Guard** : Doit capturer le prisonnier avant qu'il n'atteigne la sortie



## 📋 Prerequisites## 📋 Prérequis



1. **GAMA Platform** installed and running1. **GAMA Platform** installé et fonctionnel

2. **Python 3.8+** with pip/conda2. **Python 3.8+** avec conda/miniconda

3. **GAMA-PettingZoo** environment configured3. **Environnement gama-pettingzoo** configuré



## 🚀 Quick Start## 🚀 Installation rapide



### Installation1. Installer les dépendances :

```bash

```bashpython run_training.py install

# Install dependencies```

pip install -r requirements.txt

Ou manuellement :

# Or install core dependencies only```bash

pip install stable-baselines3 torch gymnasium matplotlib numpypip install -r requirements_training.txt

``````



### Launch GAMA Server2. Vérifier que GAMA est en marche sur le port 1001 (par défaut)



```bash## 💻 Utilisation

# Linux/macOS

./gama-headless.sh -socket 1001### Entraînement des agents



# Windows  ```bash

gama-headless.bat -socket 1001# Entraînement standard (1000 épisodes)

```python run_training.py train



### Basic Training# Entraînement rapide pour tests (50 épisodes)

python run_training.py train --config quick

```bash

# Standard training (1000 episodes)# Entraînement intensif (5000 épisodes)

python run_training.py trainpython run_training.py train --config intensive



# Quick test (50 episodes)# Entraînement personnalisé

python run_training.py train --config quickpython run_training.py train --episodes 500 --save-dir ./my_models --port 1002

```

# Custom training

python run_training.py train --episodes 500 --save-dir ./my_models### Évaluation des agents

```

```bash

## 📁 Project Structure# Évaluation standard (100 épisodes)

python run_training.py eval

```text

Prison Escape/# Évaluation avec paramètres personnalisés

├── 📄 README.md                    # This documentationpython run_training.py eval --episodes 50 --render 10 --models-dir ./my_models

├── 📄 requirements.txt             # Python dependencies

├── 📄 config.py                    # Training configurations# Évaluer un modèle spécifique

├── 📄 run_training.py              # Main training scriptpython run_training.py eval --episode-number 500

├── 📄 train_prison_escape.py       # PPO training implementation  ```

├── 📄 improved_train_prison_escape.py  # Q-Learning with anti-static mechanisms

├── 📄 evaluate_agents.py           # Agent evaluation module### Informations sur les modèles

├── 📄 improved_evaluate_agents.py  # Enhanced evaluation with metrics

├── 📄 test_environment.py          # Environment testing utilities```bash

├── 📄 image_viewer.py              # Real-time visualization tool# Afficher les modèles disponibles

├── 📄 controler.gaml               # GAMA PettingZoo controllerpython run_training.py info

├── 📄 PrisonEscape.gaml            # Main GAMA simulation model

└── 📁 snapshot/                    # Generated visualization frames# Avec un répertoire personnalisé

python run_training.py info --models-dir ./my_models

# Generated during training:```

trained_models/

├── prisoner_ppo_episode_*.zip      # Trained prisoner models## 📁 Structure des fichiers

├── guard_ppo_episode_*.zip         # Trained guard models  

├── training_metrics_*.pkl          # Training metrics data```

└── training_progress_*.png         # Progress visualizationPrison Escape/

```├── controler.gaml              # Contrôleur GAMA pour PettingZoo

├── PrisonEscape.gaml          # Modèle GAMA principal

## 🤖 Training Algorithms├── prison_petz.py             # Script de test basique

├── train_prison_escape.py     # Module d'entraînement principal

### PPO (Proximal Policy Optimization)├── evaluate_agents.py         # Module d'évaluation

- **File**: `train_prison_escape.py`├── config.py                  # Configurations d'entraînement

- **Features**: Stable, reliable training for both agents├── run_training.py            # Script utilitaire principal

- **Best for**: Standard training scenarios├── requirements_training.txt   # Dépendances Python

└── README.md                  # Ce fichier

### Improved Q-Learning

- **File**: `improved_train_prison_escape.py` # Générés lors de l'entraînement :

- **Features**: Anti-static mechanisms, curriculum learning, diversity bonusestrained_models/

- **Best for**: Avoiding convergence to static strategies├── prisoner_ppo_episode_*.zip          # Modèles entraînés prisoner

├── guard_ppo_episode_*.zip             # Modèles entraînés guard

## 💻 Usage Examples├── training_metrics_episode_*.pkl      # Métriques d'entraînement

└── training_progress_episode_*.png     # Graphiques de progression

### Training Commands```



```bash## ⚙️ Configuration

# PPO training with different configurations

python run_training.py train --config default     # 1000 episodes### Configurations prédéfinies

python run_training.py train --config quick       # 50 episodes  

python run_training.py train --config intensive   # 5000 episodes- **default** : Configuration standard (1000 épisodes)

- **quick** : Tests rapides (50 épisodes)

# Improved Q-Learning training- **intensive** : Entraînement approfondi (5000 épisodes)

python improved_train_prison_escape.py

### Paramètres principaux

# Custom parameters

python run_training.py train --episodes 2000 --port 1002| Paramètre | Description | Valeur par défaut |

```|-----------|-------------|-------------------|

| `NUM_EPISODES` | Nombre total d'épisodes | 1000 |

### Evaluation Commands| `MAX_STEPS_PER_EPISODE` | Étapes max par épisode | 200 |

| `SAVE_INTERVAL` | Sauvegarde tous les N épisodes | 100 |

```bash| `LOG_INTERVAL` | Affichage stats tous les N épisodes | 10 |

# Evaluate trained models| `GAMA_PORT` | Port de connexion GAMA | 1001 |

python run_training.py eval --episodes 100

### Paramètres PPO

# Enhanced evaluation with visualizations

python improved_evaluate_agents.py| Agent | Learning Rate | Batch Size | Epochs | Gamma |

|-------|---------------|------------|--------|-------|

# Evaluate specific models| Prisoner | 3e-4 | 64 | 10 | 0.99 |

python run_training.py eval --models-dir ./my_models --render 5| Guard | 3e-4 | 64 | 10 | 0.99 |

```

## 📊 Métriques et visualisation

### Visualization

L'entraînement génère automatiquement :

```bash

# Real-time environment visualization1. **Récompenses moyennes** par agent

python image_viewer.py2. **Taux de victoire** (prisoner vs guard)

3. **Longueur des épisodes**

# View training progress4. **Progression de l'apprentissage**

python run_training.py info --models-dir ./trained_models

```Les graphiques sont sauvegardés dans le répertoire des modèles.



## ⚙️ Configuration## 🔧 Architecture technique



### Main Parameters### Environnement

- **Grille** : 7x7 cases

| Parameter | Description | Default |- **Actions** : 4 directions (gauche, droite, haut, bas)

|-----------|-------------|---------|- **Observations** : Positions du prisoner, guard et sortie

| `NUM_EPISODES` | Total training episodes | 1000 |- **Récompenses** :

| `MAX_STEPS_PER_EPISODE` | Max steps per episode | 200 |  - Prisoner : +1 si évasion, -1 si capture

| `SAVE_INTERVAL` | Save models every N episodes | 100 |  - Guard : +1 si capture, -1 si évasion

| `LOG_INTERVAL` | Display stats every N episodes | 10 |

| `GAMA_PORT` | GAMA connection port | 1001 |### Algorithme d'apprentissage

- **PPO** (Proximal Policy Optimization)

### Algorithm Settings- **Politique** : MLP (Multi-Layer Perceptron)

- **Entraînement alterné** des deux agents

#### PPO Configuration

- **Learning Rate**: 3e-4## 🐛 Dépannage

- **Batch Size**: 64  

- **Training Epochs**: 10### Erreurs communes

- **Discount Factor**: 0.99

1. **"Impossible de se connecter à GAMA"**

#### Q-Learning Configuration     - Vérifiez que GAMA est lancé

- **Learning Rate**: 0.1   - Vérifiez le port (défaut : 1001)

- **Epsilon**: 0.3 → 0.05 (adaptive)   - Utilisez `--port` pour changer le port

- **Epsilon Decay**: 0.9995

- **Discount Factor**: 0.952. **"Module stable_baselines3 introuvable"**

   - Exécutez : `python run_training.py install`

## 🎮 Environment Details   - Ou : `pip install stable-baselines3`



### Grid Layout3. **"Erreur lors du chargement des modèles"**

- **Size**: 7x7 grid   - Vérifiez que l'entraînement s'est terminé correctement

- **Prisoner Start**: Top-left corner   - Utilisez : `python run_training.py info` pour voir les modèles disponibles

- **Guard Start**: Center of grid

- **Escape Cell**: Bottom-right corner (brown)### Performance



### Action Space- **GPU recommandé** pour l'entraînement intensif

- **0**: Stay in place- **RAM** : Minimum 8GB, recommandé 16GB

- **1**: Move up- **Temps d'entraînement** : ~2-6h selon la configuration

- **2**: Move down  

- **3**: Move left## 📈 Exemple d'utilisation complète

- **4**: Move right

```bash

### Observation Space# 1. Installation

For each agent:python run_training.py install

- Own position (x, y)

- Other agent position (x, y) # 2. Test rapide

- Escape cell position (x, y)python run_training.py train --config quick

- Grid boundaries and obstacles

# 3. Vérification des résultats

### Reward Structurepython run_training.py info

- **Prisoner**: +1 for escape, -1 for capture, 0 otherwise

- **Guard**: +1 for capture, -1 for escape, 0 otherwise# 4. Évaluation

python run_training.py eval --episodes 20 --render 3

## 📊 Training Metrics

# 5. Entraînement complet

### Tracked Metricspython run_training.py train --config default

- **Episode Rewards**: Cumulative rewards per agent

- **Win Rates**: Success percentage for each agent# 6. Évaluation finale

- **Episode Lengths**: Steps until terminationpython run_training.py eval --episodes 100

- **Learning Progress**: Policy improvement over time```



### Visualization Features## 🔄 Workflow d'entraînement

- Real-time training progress plots

- Win rate evolution charts1. **Initialisation** : Création des modèles PPO pour chaque agent

- Episode length distributions2. **Collecte** : Exécution d'épisodes et collecte des données

- Agent movement heatmaps3. **Apprentissage** : Mise à jour des politiques avec PPO

4. **Évaluation** : Test périodique des performances

## 🔧 Advanced Features5. **Sauvegarde** : Sauvegarde des modèles et métriques



### Anti-Static Mechanisms (Improved Q-Learning)## 📝 Notes importantes

- **Static Behavior Detection**: Identifies repetitive patterns

- **Dynamic Penalties**: Punishes static strategies- Les agents s'entraînent de manière **alternée**

- **Exploration Bonuses**: Rewards diverse actions- Les **récompenses sont nulles** sauf en fin d'épisode

- **Curriculum Learning**: Adaptive episode lengths- L'entraînement peut être **interrompu** et **repris**

- Les **graphiques** sont mis à jour automatiquement

### Enhanced Evaluation

- **Statistical Analysis**: Comprehensive performance metrics## 🤝 Contribution

- **Behavior Analysis**: Movement pattern detection

- **Strategy Visualization**: Action frequency heatmapsPour améliorer le système :

- **Performance Comparisons**: Multi-model evaluation1. Modifier les paramètres dans `config.py`

2. Ajuster les récompenses dans `controler.gaml`

## 🐛 Troubleshooting3. Expérimenter avec d'autres algorithmes dans `train_prison_escape.py`



### Common Issues## 📄 Licence



1. **"Cannot connect to GAMA"**Ce projet fait partie du framework GAMA et suit la même licence.

   - Ensure GAMA is running on correct port
   - Check firewall settings
   - Verify GAMA model loads correctly

2. **"Module not found" errors**  
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python environment activation

3. **Training very slow**
   - Use GPU if available (PyTorch CUDA)
   - Reduce episode count for testing
   - Check system resources (RAM/CPU)

4. **Models not saving**
   - Check directory permissions
   - Ensure sufficient disk space
   - Verify save path exists

### Performance Optimization
- **GPU Training**: Install PyTorch with CUDA support
- **Memory**: Minimum 8GB RAM, recommended 16GB
- **Training Time**: 30min-4h depending on configuration

## 📈 Example Workflow

```bash
# 1. Environment setup
pip install -r requirements.txt
./gama-headless.sh -socket 1001

# 2. Quick test training
python run_training.py train --config quick

# 3. Check results  
python run_training.py info

# 4. Full training
python improved_train_prison_escape.py  # or
python run_training.py train --config default

# 5. Comprehensive evaluation
python improved_evaluate_agents.py

# 6. Visualize performance
python image_viewer.py
```

## 🧪 Experimentation

### Hyperparameter Tuning
Modify parameters in `config.py`:
- Learning rates
- Network architectures  
- Reward structures
- Training schedules

### Algorithm Comparison
Compare different approaches:
- PPO vs Q-Learning performance
- Static vs anti-static strategies
- Various exploration strategies

### Environment Modifications
Extend the environment:
- Larger grid sizes
- Multiple prisoners/guards
- Dynamic obstacles
- Power-ups and items

## 🤝 Contributing

To improve the training system:

1. **Fork** the repository
2. **Modify** algorithms in training files
3. **Adjust** rewards in `controler.gaml`
4. **Test** with different configurations
5. **Submit** pull request with improvements

### Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive docstrings
- Include unit tests for new features
- Document algorithm modifications

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [GAMA Platform](https://gama-platform.org/)
- [PettingZoo Multi-Agent Environments](https://pettingzoo.farama.org/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## 📄 License

This project is part of the GAMA platform and follows the same MIT license.

---

🎮 **Tip**: Start with quick training to verify setup, then move to full training for best results!