# Entraînement d'agents dans Prison Escape avec GAMA

Ce projet permet d'entraîner deux agents IA (prisoner et guard) dans l'environnement Prison Escape de GAMA en utilisant l'apprentissage par renforcement avec l'algorithme PPO (Proximal Policy Optimization).

## 🎯 Objectif

- **Prisoner** : Doit atteindre la case d'évasion (brune) sans se faire capturer
- **Guard** : Doit capturer le prisonnier avant qu'il n'atteigne la sortie

## 📋 Prérequis

1. **GAMA Platform** installé et fonctionnel
2. **Python 3.8+** avec conda/miniconda
3. **Environnement gama-pettingzoo** configuré

## 🚀 Installation rapide

1. Installer les dépendances :
```bash
python run_training.py install
```

Ou manuellement :
```bash
pip install -r requirements_training.txt
```

2. Vérifier que GAMA est en marche sur le port 1001 (par défaut)

## 💻 Utilisation

### Entraînement des agents

```bash
# Entraînement standard (1000 épisodes)
python run_training.py train

# Entraînement rapide pour tests (50 épisodes)
python run_training.py train --config quick

# Entraînement intensif (5000 épisodes)
python run_training.py train --config intensive

# Entraînement personnalisé
python run_training.py train --episodes 500 --save-dir ./my_models --port 1002
```

### Évaluation des agents

```bash
# Évaluation standard (100 épisodes)
python run_training.py eval

# Évaluation avec paramètres personnalisés
python run_training.py eval --episodes 50 --render 10 --models-dir ./my_models

# Évaluer un modèle spécifique
python run_training.py eval --episode-number 500
```

### Informations sur les modèles

```bash
# Afficher les modèles disponibles
python run_training.py info

# Avec un répertoire personnalisé
python run_training.py info --models-dir ./my_models
```

## 📁 Structure des fichiers

```
Prison Escape/
├── controler.gaml              # Contrôleur GAMA pour PettingZoo
├── PrisonEscape.gaml          # Modèle GAMA principal
├── prison_petz.py             # Script de test basique
├── train_prison_escape.py     # Module d'entraînement principal
├── evaluate_agents.py         # Module d'évaluation
├── config.py                  # Configurations d'entraînement
├── run_training.py            # Script utilitaire principal
├── requirements_training.txt   # Dépendances Python
└── README.md                  # Ce fichier

# Générés lors de l'entraînement :
trained_models/
├── prisoner_ppo_episode_*.zip          # Modèles entraînés prisoner
├── guard_ppo_episode_*.zip             # Modèles entraînés guard
├── training_metrics_episode_*.pkl      # Métriques d'entraînement
└── training_progress_episode_*.png     # Graphiques de progression
```

## ⚙️ Configuration

### Configurations prédéfinies

- **default** : Configuration standard (1000 épisodes)
- **quick** : Tests rapides (50 épisodes)
- **intensive** : Entraînement approfondi (5000 épisodes)

### Paramètres principaux

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `NUM_EPISODES` | Nombre total d'épisodes | 1000 |
| `MAX_STEPS_PER_EPISODE` | Étapes max par épisode | 200 |
| `SAVE_INTERVAL` | Sauvegarde tous les N épisodes | 100 |
| `LOG_INTERVAL` | Affichage stats tous les N épisodes | 10 |
| `GAMA_PORT` | Port de connexion GAMA | 1001 |

### Paramètres PPO

| Agent | Learning Rate | Batch Size | Epochs | Gamma |
|-------|---------------|------------|--------|-------|
| Prisoner | 3e-4 | 64 | 10 | 0.99 |
| Guard | 3e-4 | 64 | 10 | 0.99 |

## 📊 Métriques et visualisation

L'entraînement génère automatiquement :

1. **Récompenses moyennes** par agent
2. **Taux de victoire** (prisoner vs guard)
3. **Longueur des épisodes**
4. **Progression de l'apprentissage**

Les graphiques sont sauvegardés dans le répertoire des modèles.

## 🔧 Architecture technique

### Environnement
- **Grille** : 7x7 cases
- **Actions** : 4 directions (gauche, droite, haut, bas)
- **Observations** : Positions du prisoner, guard et sortie
- **Récompenses** :
  - Prisoner : +1 si évasion, -1 si capture
  - Guard : +1 si capture, -1 si évasion

### Algorithme d'apprentissage
- **PPO** (Proximal Policy Optimization)
- **Politique** : MLP (Multi-Layer Perceptron)
- **Entraînement alterné** des deux agents

## 🐛 Dépannage

### Erreurs communes

1. **"Impossible de se connecter à GAMA"**
   - Vérifiez que GAMA est lancé
   - Vérifiez le port (défaut : 1001)
   - Utilisez `--port` pour changer le port

2. **"Module stable_baselines3 introuvable"**
   - Exécutez : `python run_training.py install`
   - Ou : `pip install stable-baselines3`

3. **"Erreur lors du chargement des modèles"**
   - Vérifiez que l'entraînement s'est terminé correctement
   - Utilisez : `python run_training.py info` pour voir les modèles disponibles

### Performance

- **GPU recommandé** pour l'entraînement intensif
- **RAM** : Minimum 8GB, recommandé 16GB
- **Temps d'entraînement** : ~2-6h selon la configuration

## 📈 Exemple d'utilisation complète

```bash
# 1. Installation
python run_training.py install

# 2. Test rapide
python run_training.py train --config quick

# 3. Vérification des résultats
python run_training.py info

# 4. Évaluation
python run_training.py eval --episodes 20 --render 3

# 5. Entraînement complet
python run_training.py train --config default

# 6. Évaluation finale
python run_training.py eval --episodes 100
```

## 🔄 Workflow d'entraînement

1. **Initialisation** : Création des modèles PPO pour chaque agent
2. **Collecte** : Exécution d'épisodes et collecte des données
3. **Apprentissage** : Mise à jour des politiques avec PPO
4. **Évaluation** : Test périodique des performances
5. **Sauvegarde** : Sauvegarde des modèles et métriques

## 📝 Notes importantes

- Les agents s'entraînent de manière **alternée**
- Les **récompenses sont nulles** sauf en fin d'épisode
- L'entraînement peut être **interrompu** et **repris**
- Les **graphiques** sont mis à jour automatiquement

## 🤝 Contribution

Pour améliorer le système :
1. Modifier les paramètres dans `config.py`
2. Ajuster les récompenses dans `controler.gaml`
3. Expérimenter avec d'autres algorithmes dans `train_prison_escape.py`

## 📄 Licence

Ce projet fait partie du framework GAMA et suit la même licence.
