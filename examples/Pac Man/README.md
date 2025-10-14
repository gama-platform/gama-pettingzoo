# Pac Man - Environnement multi-agents avec GAMA-PettingZoo

Version multi-agents du célèbre jeu Pac-Man implémentée avec GAMA-PettingZoo. Cet environnement permet d'entraîner des agents dans un jeu de coopération/compétition complexe.

## 🎯 Objectif

- **Pac-Man** : Collecter tous les points tout en évitant les fantômes
- **Fantômes** : Capturer Pac-Man ou l'empêcher de collecter les points
- **Coopération/Compétition** : Apprentissage de stratégies multi-agents

## 📋 Description de l'environnement

### Agents
- **1 Pac-Man** : Agent principal collecteur
- **2-4 Fantômes** : Agents antagonistes
- **Interactions** : Capture, évitement, poursuite

### Mécaniques de jeu
- **Labyrinthe** : Navigation dans un maze avec murs
- **Points** : Collecte pour gagner des récompenses
- **Power-ups** : Bonus temporaires (optionnel)
- **Collision** : Pac-Man vs Fantômes détermine la fin

## 🚀 Démarrage rapide

### Prérequis

1. **GAMA Platform** installé et configuré
2. **Python 3.8+** avec dépendances :
   ```bash
   pip install gama-pettingzoo numpy matplotlib stable-baselines3
   ```

### Lancement

1. **Démarrer GAMA** en mode serveur :
   ```bash
   # Linux/MacOS
   ./gama-headless.sh -socket 1001
   
   # Windows
   gama-headless.bat -socket 1001
   ```

2. **Lancer le jeu** :
   ```bash
   python pacman_petz.py
   ```

## 📁 Structure des fichiers

```text
Pac Man/
├── PacMan.gaml            # Modèle GAMA principal du jeu
├── pacman_petz.py         # Script Python d'entraînement/jeu
└── README.md             # Cette documentation

# Générés lors de l'entraînement :
trained_models/
├── pacman_*.zip          # Modèles Pac-Man entraînés
├── ghost_*.zip           # Modèles fantômes entraînés
└── metrics/              # Métriques d'entraînement
```

## ⚙️ Configuration

### Paramètres GAMA (PacMan.gaml)

- **`maze_size`** : Dimensions du labyrinthe
- **`num_ghosts`** : Nombre de fantômes (2-4)
- **`num_dots`** : Nombre de points à collecter
- **`ghost_speed`** : Vitesse relative des fantômes

### Paramètres Python (pacman_petz.py)

- **Algorithmes** : PPO, DQN, MADDPG
- **Episodes d'entraînement** : 1000-5000
- **Stratégies** : Coopératives ou compétitives

## 🎮 Modes de jeu

### 1. Mode Test/Visualisation
```python
python pacman_petz.py --mode test --render
```

### 2. Mode Entraînement
```python
python pacman_petz.py --mode train --episodes 2000
```

### 3. Mode Évaluation
```python
python pacman_petz.py --mode eval --models-dir ./trained_models
```

## 🤖 Algorithmes d'apprentissage

### Recommandés pour cet environnement

1. **PPO (Proximal Policy Optimization)**
   - **Usage** : Entraînement stable pour tous les agents
   - **Avantages** : Convergence robuste, hyperparamètres faciles

2. **MADDPG (Multi-Agent DDPG)**
   - **Usage** : Environnements compétitifs complexes
   - **Avantages** : Gestion des agents multiples, politiques continues

3. **Independent Q-Learning**
   - **Usage** : Apprentissage simple et rapide
   - **Avantages** : Facile à implémenter, bon pour débuter

## 📊 Espaces d'actions et observations

### Actions (Discrete)
- **0** : Reste immobile
- **1** : Haut
- **2** : Bas  
- **3** : Gauche
- **4** : Droite

### Observations (Box)
Pour chaque agent :
- **Position propre** : (x, y)
- **Positions autres agents** : [(x1, y1), (x2, y2), ...]
- **Points visibles** : Positions des points proches
- **Murs détectés** : Information de navigation

## 🏆 Stratégies d'entraînement

### Pac-Man
- **Récompenses** : +1 par point collecté, +10 si tous collectés, -10 si capturé
- **Stratégie** : Navigation efficace, évitement des fantômes
- **Difficultés** : Équilibrer collecte et survie

### Fantômes
- **Récompenses** : +10 si Pac-Man capturé, -1 par point collecté par Pac-Man
- **Stratégie** : Coordination, encerclement, prédiction
- **Difficultés** : Éviter les blocages, coopération

## 📈 Métriques de performance

### Métriques de jeu
- **Score Pac-Man** : Points collectés par partie
- **Taux de capture** : Victoires des fantômes (%)
- **Durée des parties** : Nombre de steps moyen
- **Efficacité** : Score/temps

### Métriques d'apprentissage
- **Récompense cumulative** par agent
- **Taux de convergence** des algorithmes
- **Stabilité** des politiques

## 🔧 Personnalisation

### Modifier la difficulté
1. **Ajuster le nombre de fantômes** dans PacMan.gaml
2. **Changer la vitesse** des agents
3. **Modifier les récompenses** dans le contrôleur
4. **Ajouter des power-ups** ou mécaniques spéciales

### Nouvelles variantes
- **Pac-Man coopératif** : Plusieurs Pac-Man
- **Fantômes avec rôles** : Chasseur, bloqueur, patrouilleur
- **Labyrinthe dynamique** : Murs qui changent
- **Mode survie** : Temps limité

## 🐛 Dépannage

### Problèmes fréquents

1. **"Agents bloqués dans les murs"**
   - Vérifiez la logique de navigation dans PacMan.gaml
   - Ajustez les règles de mouvement

2. **"Apprentissage très lent"**
   - Réduisez la complexité du labyrinthe
   - Ajustez les hyperparamètres d'apprentissage
   - Utilisez des récompenses intermédiaires

3. **"Fantômes ne coopèrent pas"**
   - Implémentez des récompenses partagées
   - Utilisez MADDPG au lieu de DQN indépendant
   - Ajoutez de la communication entre agents

## 🎯 Exercices suggérés

### Niveau débutant
1. **Pac-Man simple** : Un seul fantôme, petit labyrinthe
2. **Navigation de base** : Collecter les points sans fantômes
3. **Évitement simple** : Fantôme avec mouvement prévisible

### Niveau intermédiaire
1. **Multi-fantômes** : 2-3 fantômes avec stratégies différentes
2. **Optimisation de chemin** : Plus court chemin vers tous les points
3. **Adaptabilité** : Fantômes qui apprennent les patterns de Pac-Man

### Niveau avancé
1. **Coordination complexe** : Encerclement coordonné
2. **Méta-apprentissage** : Adaptation rapide à nouveaux labyrinthes
3. **Communication** : Échange d'informations entre fantômes

## 🔗 Ressources

- [Pac-Man AI Research](https://inst.eecs.berkeley.edu/~cs188/sp19/project2.html)
- [Multi-Agent RL Algorithms](https://github.com/oxwhirl/smac)
- [MADDPG Paper](https://arxiv.org/abs/1706.02275)

## 🤝 Contribution

Améliorations bienvenues :
- Nouveaux algorithmes d'entraînement
- Variantes de jeu intéressantes
- Métriques d'évaluation avancées
- Visualisations en temps réel

---

🎮 **Conseil** : Commencez par entraîner Pac-Man seul, puis ajoutez progressivement les fantômes !