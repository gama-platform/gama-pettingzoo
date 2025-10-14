# Moving Exemple - Agents mobiles avec GAMA-PettingZoo

Exemple d'introduction aux agents mobiles utilisant GAMA-PettingZoo. Cet environnement simple permet d'apprendre les concepts de base de l'apprentissage par renforcement multi-agents.

## 🎯 Objectif

Créer et entraîner des agents capables de naviguer dans un environnement 2D en évitant les collisions et en atteignant des objectifs.

## 📋 Description de l'environnement

- **Grille** : Environnement 2D configurable
- **Agents** : Un ou plusieurs agents mobiles
- **Actions** : Déplacement dans 4 directions (haut, bas, gauche, droite)
- **Objectifs** : Navigation autonome et coordination entre agents

## 🚀 Démarrage rapide

### Prérequis

1. **GAMA Platform** installé et configuré
2. **Python 3.8+** avec les dépendances :
   ```bash
   pip install gama-pettingzoo numpy matplotlib
   ```

### Lancement

1. **Démarrer GAMA** en mode serveur :
   ```bash
   # Linux/MacOS
   ./gama-headless.sh -socket 1001
   
   # Windows
   gama-headless.bat -socket 1001
   ```

2. **Lancer la simulation** :
   ```bash
   python image_viewer.py
   ```

## 📁 Structure des fichiers

```text
Moving Exemple/
├── MovingEx.gaml           # Modèle GAMA principal
├── image_viewer.py         # Visualiseur et contrôleur Python
└── README.md              # Cette documentation
```

## ⚙️ Configuration

### Paramètres GAMA (MovingEx.gaml)

- **`grid_size`** : Taille de la grille (par défaut : 20x20)
- **`num_agents`** : Nombre d'agents mobiles
- **`step_duration`** : Durée de chaque étape de simulation

### Paramètres Python (image_viewer.py)

- **Port GAMA** : 1001 (configurable)
- **Taux de rafraîchissement** : Configurable pour la visualisation

## 🎮 Utilisation

### Interface de base

L'`image_viewer.py` fournit :
- **Visualisation** en temps réel des agents
- **Contrôle** de la simulation (pause/play)
- **Statistiques** de performance

### Personnalisation

Vous pouvez modifier :
1. **Le modèle GAMA** pour changer l'environnement
2. **Le script Python** pour différents algorithmes d'apprentissage
3. **Les paramètres** pour ajuster la difficulté

## 🔧 Extension de l'exemple

### Ajouter l'apprentissage par renforcement

```python
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

# Créer l'environnement
env = GamaParallelEnv(
    gaml_experiment_path='MovingEx.gaml',
    gaml_experiment_name='main',
    gama_ip_address='localhost',
    gama_port=1001
)

# Boucle d'entraînement basique
observations, infos = env.reset()
for step in range(1000):
    actions = {agent: env.action_space(agent).sample() 
              for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if all(terminations.values()) or all(truncations.values()):
        observations, infos = env.reset()
```

### Algorithmes suggérés

1. **Q-Learning** : Pour débuter avec un agent
2. **DQN** : Apprentissage profond simple
3. **PPO multi-agents** : Pour plusieurs agents coopératifs
4. **MADDPG** : Pour des environnements complexes

## 📊 Métriques de performance

Mesurez :
- **Temps de convergence** vers l'objectif
- **Nombre de collisions** entre agents
- **Efficacité de la navigation** (distance parcourue)
- **Coordination** entre agents multiples

## 🐛 Dépannage

### Problèmes courants

1. **"Connexion à GAMA échouée"**
   - Vérifiez que GAMA est lancé avec le bon port
   - Vérifiez les paramètres de connexion

2. **"Erreur de visualisation"**
   - Installez matplotlib : `pip install matplotlib`
   - Vérifiez les permissions d'affichage

3. **"Modèle GAMA ne répond pas"**
   - Redémarrez GAMA
   - Vérifiez la syntaxe du fichier .gaml

## 🎯 Exercices suggérés

### Niveau débutant
1. **Agent unique** : Faire naviguer un agent vers un point fixe
2. **Évitement d'obstacles** : Ajouter des obstacles statiques
3. **Objectifs multiples** : Plusieurs points à visiter

### Niveau intermédiaire
1. **Agents multiples** : Coordination de 2-3 agents
2. **Évitement de collisions** : Entre agents mobiles
3. **Objectifs dynamiques** : Points qui bougent

### Niveau avancé
1. **Formation d'agents** : Maintenir une formation géométrique
2. **Optimisation de chemin** : Trouver le chemin optimal
3. **Adaptation** : Environnement qui change

## 🔗 Liens utiles

- [GAMA Platform](https://gama-platform.org/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Reinforcement Learning Introduction](https://spinningup.openai.com/en/latest/)

## 🤝 Contribution

Améliorations suggérées :
- Ajouter plus d'algorithmes d'exemple
- Créer des visualisations avancées
- Implémenter des métriques automatiques
- Ajouter des configurations prédéfinies

---

💡 **Conseil** : Commencez par comprendre le modèle GAMA avant d'ajouter l'apprentissage automatique !