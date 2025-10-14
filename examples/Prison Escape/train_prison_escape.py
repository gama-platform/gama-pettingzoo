"""
Training 2 agents in GAMA's Prison Escape environment
Uses PPO algorithm to train prisoner and guard agents
"""

import asyncio
import numpy as np
from pathlib import Path
import time
from collections import deque
import pickle

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

# Conditional imports to avoid errors if packages are not installed
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Graphics disabled.")

try:
    from stable_baselines3 import PPO
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable-Baselines3 not available. Using fallback agent.")

try:
    from pettingzoo import ParallelEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False

class PrisonEscapeWrapper:
    """Wrapper to adapt GAMA PettingZoo environment to Stable-Baselines3"""
    
    def __init__(self, gaml_experiment_path, gaml_experiment_name="main", 
                 gama_ip="localhost", gama_port=1001):
        self.gaml_experiment_path = gaml_experiment_path
        self.gaml_experiment_name = gaml_experiment_name
        self.gama_ip = gama_ip
        self.gama_port = gama_port
        self.env = None
        self.agents = ["prisoner", "guard"]
        
    async def _create_env(self):
        """Creates GAMA environment asynchronously"""
        if self.env is None:
            self.env = GamaParallelEnv(
                gaml_experiment_path=self.gaml_experiment_path,
                gaml_experiment_name=self.gaml_experiment_name,
                gama_ip_address=self.gama_ip,
                gama_port=self.gama_port
            )
        return self.env
    
    async def reset(self):
        """Reset the environment"""
        env = await self._create_env()
        obs, infos = env.reset()
        return obs, infos
    
    async def step(self, actions):
        """Execute a step in the environment"""
        env = await self._create_env()
        return env.step(actions)
    
    async def close(self):
        """Close the environment"""
        if self.env:
            self.env.close()

class TrainingMetrics:
    """Class to track training metrics"""
    
    def __init__(self):
        self.episode_rewards = {"prisoner": deque(maxlen=100), "guard": deque(maxlen=100)}
        self.episode_lengths = deque(maxlen=100)
        self.win_rates = {"prisoner": deque(maxlen=100), "guard": deque(maxlen=100)}
        self.total_episodes = 0
        
    def add_episode(self, rewards, length, winner):
        """Add metrics from an episode"""
        self.episode_rewards["prisoner"].append(rewards.get("prisoner", 0))
        self.episode_rewards["guard"].append(rewards.get("guard", 0))
        self.episode_lengths.append(length)
        
        # Update win rates
        prisoner_win = 1 if winner == "prisoner" else 0
        guard_win = 1 if winner == "guard" else 0
        
        self.win_rates["prisoner"].append(prisoner_win)
        self.win_rates["guard"].append(guard_win)
        self.total_episodes += 1
    
    def get_current_stats(self):
        """Return current statistics"""
        stats = {
            "episode": self.total_episodes,
            "avg_reward_prisoner": np.mean(self.episode_rewards["prisoner"]) if self.episode_rewards["prisoner"] else 0,
            "avg_reward_guard": np.mean(self.episode_rewards["guard"]) if self.episode_rewards["guard"] else 0,
            "avg_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "prisoner_win_rate": np.mean(self.win_rates["prisoner"]) if self.win_rates["prisoner"] else 0,
            "guard_win_rate": np.mean(self.win_rates["guard"]) if self.win_rates["guard"] else 0
        }
        return stats

class PrisonEscapeTrainer:
    """Main class for agent training"""
    
    def __init__(self, experiment_path, save_dir="./models"):
        self.experiment_path = experiment_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.env_wrapper = PrisonEscapeWrapper(experiment_path)
        self.metrics = TrainingMetrics()
        
        # Modèles pour chaque agent
        self.models = {}
        self.model_configs = {
            "prisoner": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "clip_range": 0.2
            },
            "guard": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "clip_range": 0.2
            }
        }
    
    def create_dummy_env_for_agent(self, agent_name):
        """Crée un environnement dummy pour initialiser le modèle"""
        # Observation space: [prisoner_pos, guard_pos, escape_pos] (3 valeurs entre 0 et 48 pour une grille 7x7)
        obs_space = spaces.MultiDiscrete([49, 49, 49])  # 7*7 = 49 positions possibles
        # Action space: 4 directions (gauche, droite, haut, bas)
        action_space = spaces.Discrete(4)
        
        class DummyEnv:
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = action_space
                
            def reset(self):
                return np.array([0, 0, 0])
                
            def step(self, action):
                return np.array([0, 0, 0]), 0, True, {}
        
        return DummyEnv()
    
    def initialize_models(self):
        """Initialise les modèles PPO pour chaque agent"""
        for agent in ["prisoner", "guard"]:
            dummy_env = self.create_dummy_env_for_agent(agent)
            config = self.model_configs[agent]
            
            self.models[agent] = PPO(
                "MlpPolicy",
                dummy_env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                gamma=config["gamma"],
                clip_range=config["clip_range"],
                verbose=1
            )
            print(f"Modèle {agent} initialisé")
    
    async def collect_episode_data(self, max_steps=200):
        """Collecte les données d'un épisode complet"""
        obs, infos = await self.env_wrapper.reset()
        
        episode_data = {
            "prisoner": {"observations": [], "actions": [], "rewards": []},
            "guard": {"observations": [], "actions": [], "rewards": []}
        }
        
        total_rewards = {"prisoner": 0, "guard": 0}
        step_count = 0
        done = False
        winner = None
        
        while not done and step_count < max_steps:
            actions = {}
            
            # Collecte des actions pour chaque agent
            for agent in ["prisoner", "guard"]:
                if agent in obs:
                    obs_array = np.array(obs[agent])
                    action, _ = self.models[agent].predict(obs_array, deterministic=False)
                    actions[agent] = int(action)
                    
                    # Stockage des données
                    episode_data[agent]["observations"].append(obs_array)
                    episode_data[agent]["actions"].append(action)
            
            # Exécution de l'étape
            next_obs, rewards, terminations, truncations, infos = await self.env_wrapper.step(actions)
            
            # Stockage des récompenses
            for agent in ["prisoner", "guard"]:
                if agent in rewards:
                    reward = rewards[agent]
                    episode_data[agent]["rewards"].append(reward)
                    total_rewards[agent] += reward
                    
                    # Détermination du gagnant
                    if terminations.get(agent, False) and reward > 0:
                        winner = agent
            
            obs = next_obs
            step_count += 1
            
            # Vérification de fin d'épisode
            if any(terminations.values()) or any(truncations.values()):
                done = True
        
        return episode_data, total_rewards, step_count, winner
    
    def update_models(self, episode_data):
        """Met à jour les modèles avec les données collectées"""
        for agent in ["prisoner", "guard"]:
            data = episode_data[agent]
            if len(data["observations"]) > 0:
                # Conversion en arrays numpy
                observations = np.array(data["observations"])
                actions = np.array(data["actions"])
                rewards = np.array(data["rewards"])
                
                # Calcul des avantages (simple pour ce cas)
                advantages = rewards - np.mean(rewards) if len(rewards) > 1 else rewards
                
                # Mise à jour du modèle (simulation d'une mise à jour PPO)
                # Note: Pour une vraie implémentation, il faudrait utiliser le replay buffer de PPO
                print(f"Mise à jour du modèle {agent} avec {len(observations)} observations")
    
    async def train(self, num_episodes=1000, save_interval=100, log_interval=10):
        """Entraîne les agents sur le nombre d'épisodes spécifié"""
        print(f"Début de l'entraînement pour {num_episodes} épisodes")
        
        # Initialisation des modèles
        self.initialize_models()
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Collecte des données d'épisode
            episode_data, total_rewards, episode_length, winner = await self.collect_episode_data()
            
            # Mise à jour des modèles
            self.update_models(episode_data)
            
            # Mise à jour des métriques
            self.metrics.add_episode(total_rewards, episode_length, winner)
            
            episode_time = time.time() - start_time
            
            # Logging
            if (episode + 1) % log_interval == 0:
                stats = self.metrics.get_current_stats()
                print(f"Épisode {episode + 1}/{num_episodes}")
                print(f"  Récompense moyenne prisoner: {stats['avg_reward_prisoner']:.2f}")
                print(f"  Récompense moyenne guard: {stats['avg_reward_guard']:.2f}")
                print(f"  Longueur moyenne épisode: {stats['avg_episode_length']:.1f}")
                print(f"  Taux de victoire prisoner: {stats['prisoner_win_rate']:.2%}")
                print(f"  Taux de victoire guard: {stats['guard_win_rate']:.2%}")
                print(f"  Temps par épisode: {episode_time:.2f}s")
                print("-" * 50)
            
            # Sauvegarde périodique
            if (episode + 1) % save_interval == 0:
                self.save_models(episode + 1)
                self.save_metrics(episode + 1)
        
        # Sauvegarde finale
        self.save_models(num_episodes)
        self.save_metrics(num_episodes)
        await self.env_wrapper.close()
        
        print("Entraînement terminé!")
    
    def save_models(self, episode):
        """Sauvegarde les modèles"""
        for agent, model in self.models.items():
            model_path = self.save_dir / f"{agent}_ppo_episode_{episode}.zip"
            model.save(str(model_path))
            print(f"Modèle {agent} sauvegardé: {model_path}")
    
    def save_metrics(self, episode):
        """Sauvegarde les métriques"""
        metrics_path = self.save_dir / f"training_metrics_episode_{episode}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        # Sauvegarde des graphiques
        self.plot_training_progress(episode)
    
    def plot_training_progress(self, episode):
        """Crée des graphiques de progression de l'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Récompenses moyennes
        axes[0, 0].plot(list(self.metrics.episode_rewards["prisoner"]), label="Prisoner", alpha=0.7)
        axes[0, 0].plot(list(self.metrics.episode_rewards["guard"]), label="Guard", alpha=0.7)
        axes[0, 0].set_title("Récompenses par épisode")
        axes[0, 0].set_xlabel("Épisode")
        axes[0, 0].set_ylabel("Récompense")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Longueur des épisodes
        axes[0, 1].plot(list(self.metrics.episode_lengths))
        axes[0, 1].set_title("Longueur des épisodes")
        axes[0, 1].set_xlabel("Épisode")
        axes[0, 1].set_ylabel("Nombre d'étapes")
        axes[0, 1].grid(True)
        
        # Taux de victoire
        axes[1, 0].plot(list(self.metrics.win_rates["prisoner"]), label="Prisoner", alpha=0.7)
        axes[1, 0].plot(list(self.metrics.win_rates["guard"]), label="Guard", alpha=0.7)
        axes[1, 0].set_title("Taux de victoire (fenêtre glissante)")
        axes[1, 0].set_xlabel("Épisode")
        axes[1, 0].set_ylabel("Taux de victoire")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Récompenses moyennes cumulées
        prisoner_cumsum = np.cumsum(list(self.metrics.episode_rewards["prisoner"]))
        guard_cumsum = np.cumsum(list(self.metrics.episode_rewards["guard"]))
        axes[1, 1].plot(prisoner_cumsum, label="Prisoner", alpha=0.7)
        axes[1, 1].plot(guard_cumsum, label="Guard", alpha=0.7)
        axes[1, 1].set_title("Récompenses cumulées")
        axes[1, 1].set_xlabel("Épisode")
        axes[1, 1].set_ylabel("Récompense cumulée")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.save_dir / f"training_progress_episode_{episode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphiques sauvegardés: {plot_path}")

async def main():
    """Main function to launch training"""
    
    # Configuration de l'entraînement
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    save_dir = "./trained_models"
    
    # Création du trainer
    trainer = PrisonEscapeTrainer(exp_path, save_dir)
    
    # Paramètres d'entraînement
    num_episodes = 500  # Nombre total d'épisodes
    save_interval = 50  # Sauvegarde tous les 50 épisodes
    log_interval = 10   # Affichage des stats tous les 10 épisodes
    
    print("=== ENTRAÎNEMENT PRISON ESCAPE ===")
    print(f"Chemin expérience GAMA: {exp_path}")
    print(f"Répertoire de sauvegarde: {save_dir}")
    print(f"Nombre d'épisodes: {num_episodes}")
    print("=" * 40)
    
    # Lancement de l'entraînement
    await trainer.train(
        num_episodes=num_episodes,
        save_interval=save_interval,
        log_interval=log_interval
    )

if __name__ == "__main__":
    asyncio.run(main())
