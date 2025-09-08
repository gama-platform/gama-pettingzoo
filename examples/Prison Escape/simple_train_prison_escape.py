"""
Agent simple de remplacement pour l'entraînement Prison Escape
Utilise une stratégie basique sans dépendances externes lourdes
"""

import asyncio
import numpy as np
from pathlib import Path
import time
import json
from collections import deque

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

class SimpleAgent:
    """Agent simple utilisant une stratégie heuristique avec apprentissage basique"""
    
    def __init__(self, agent_name, grid_size=7):
        self.agent_name = agent_name
        self.grid_size = grid_size
        self.action_count = 4  # gauche, droite, haut, bas
        
        # Table Q simple (état -> action -> valeur)
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 1.0  # Exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Facteur de discount
        
        self.last_state = None
        self.last_action = None
        
    def get_state_key(self, observation):
        """Convertit l'observation en clé d'état"""
        return tuple(observation)
    
    def get_action(self, observation, training=True):
        """Choisit une action basée sur l'observation"""
        state_key = self.get_state_key(observation)
        
        # Initialisation de l'état si nouveau
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_count)
        
        if training and np.random.random() < self.epsilon:
            # Exploration : action aléatoire
            action = np.random.randint(0, self.action_count)
        else:
            # Exploitation : meilleure action connue
            action = np.argmax(self.q_table[state_key])
        
        self.last_state = state_key
        self.last_action = action
        
        return action
    
    def update(self, reward, next_observation, done):
        """Met à jour la table Q"""
        if self.last_state is None or self.last_action is None:
            return
        
        next_state_key = self.get_state_key(next_observation)
        
        # Initialisation du nouvel état si nécessaire
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_count)
        
        # Calcul de la valeur Q mise à jour
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Mise à jour de la table Q
        current_q = self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] = current_q + self.learning_rate * (target - current_q)
        
        # Décroissance de l'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Sauvegarde l'agent"""
        data = {
            'agent_name': self.agent_name,
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Agent {self.agent_name} sauvegardé: {filepath}")
    
    def load(self, filepath):
        """Charge l'agent"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.agent_name = data['agent_name']
        self.q_table = {eval(k): np.array(v) for k, v in data['q_table'].items()}
        self.epsilon = data['epsilon']
        self.learning_rate = data['learning_rate']
        self.gamma = data['gamma']
        print(f"Agent {self.agent_name} chargé: {filepath}")

class SimpleTrainingMetrics:
    """Métriques d'entraînement simplifiées"""
    
    def __init__(self):
        self.episode_rewards = {"prisoner": [], "guard": []}
        self.episode_lengths = []
        self.win_rates = {"prisoner": [], "guard": []}
        self.total_episodes = 0
        self.recent_window = 100
        
    def add_episode(self, rewards, length, winner):
        """Ajoute les métriques d'un épisode"""
        self.episode_rewards["prisoner"].append(rewards.get("prisoner", 0))
        self.episode_rewards["guard"].append(rewards.get("guard", 0))
        self.episode_lengths.append(length)
        
        # Calcul du taux de victoire sur une fenêtre glissante
        prisoner_win = 1 if winner == "prisoner" else 0
        guard_win = 1 if winner == "guard" else 0
        
        self.win_rates["prisoner"].append(prisoner_win)
        self.win_rates["guard"].append(guard_win)
        
        # Maintenir seulement les dernières entrées
        if len(self.win_rates["prisoner"]) > self.recent_window:
            self.win_rates["prisoner"].pop(0)
            self.win_rates["guard"].pop(0)
        
        self.total_episodes += 1
    
    def get_current_stats(self):
        """Retourne les statistiques actuelles"""
        recent_prisoner = self.episode_rewards["prisoner"][-self.recent_window:]
        recent_guard = self.episode_rewards["guard"][-self.recent_window:]
        recent_lengths = self.episode_lengths[-self.recent_window:]
        
        stats = {
            "episode": self.total_episodes,
            "avg_reward_prisoner": np.mean(recent_prisoner) if recent_prisoner else 0,
            "avg_reward_guard": np.mean(recent_guard) if recent_guard else 0,
            "avg_episode_length": np.mean(recent_lengths) if recent_lengths else 0,
            "prisoner_win_rate": np.mean(self.win_rates["prisoner"]) if self.win_rates["prisoner"] else 0,
            "guard_win_rate": np.mean(self.win_rates["guard"]) if self.win_rates["guard"] else 0,
            "epsilon_prisoner": 0,  # Sera mis à jour par le trainer
            "epsilon_guard": 0
        }
        return stats
    
    def save(self, filepath):
        """Sauvegarde les métriques"""
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "win_rates": self.win_rates,
            "total_episodes": self.total_episodes
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

class SimplePrisonEscapeTrainer:
    """Trainer simplifié pour Prison Escape"""
    
    def __init__(self, experiment_path, save_dir="./simple_models"):
        self.experiment_path = experiment_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.env_wrapper = None
        self.metrics = SimpleTrainingMetrics()
        
        # Création des agents
        self.agents = {
            "prisoner": SimpleAgent("prisoner"),
            "guard": SimpleAgent("guard")
        }
        
        print(f"Trainer simplifié initialisé. Répertoire: {self.save_dir}")
    
    def create_env(self):
        """Crée l'environnement GAMA"""
        if self.env_wrapper is None:
            self.env_wrapper = GamaParallelEnv(
                gaml_experiment_path=self.experiment_path,
                gaml_experiment_name="main",
                gama_ip_address="localhost",
                gama_port=1001
            )
        return self.env_wrapper
    
    def run_episode(self, max_steps=200, training=True):
        """Exécute un épisode complet"""
        env = self.create_env()
        obs, infos = env.reset()
        
        total_rewards = {"prisoner": 0, "guard": 0}
        step_count = 0
        done = False
        winner = None
        
        # États précédents pour l'apprentissage
        prev_obs = {}
        
        while not done and step_count < max_steps:
            actions = {}
            
            # Collecte des actions pour chaque agent
            for agent_name in ["prisoner", "guard"]:
                if agent_name in obs:
                    obs_array = np.array(obs[agent_name])
                    action = self.agents[agent_name].get_action(obs_array, training)
                    actions[agent_name] = int(action)
                    prev_obs[agent_name] = obs_array
            
            # Exécution de l'étape
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Mise à jour des agents si en mode entraînement
            if training:
                for agent_name in ["prisoner", "guard"]:
                    if agent_name in rewards and agent_name in prev_obs:
                        reward = rewards[agent_name]
                        next_obs_array = np.array(next_obs.get(agent_name, prev_obs[agent_name]))
                        is_done = terminations.get(agent_name, False) or truncations.get(agent_name, False)
                        
                        self.agents[agent_name].update(reward, next_obs_array, is_done)
                        total_rewards[agent_name] += reward
                        
                        # Détermination du gagnant
                        if is_done and reward > 0:
                            winner = agent_name
            
            obs = next_obs
            step_count += 1
            
            # Vérification de fin d'épisode
            if any(terminations.values()) or any(truncations.values()):
                done = True
        
        return total_rewards, step_count, winner
    
    def train(self, num_episodes=500, save_interval=50, log_interval=10):
        """Entraîne les agents"""
        print(f"=== DÉBUT DE L'ENTRAÎNEMENT ({num_episodes} épisodes) ===")
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Exécution d'un épisode d'entraînement
            total_rewards, episode_length, winner = self.run_episode(training=True)
            
            # Mise à jour des métriques
            self.metrics.add_episode(total_rewards, episode_length, winner)
            
            episode_time = time.time() - start_time
            
            # Logging périodique
            if (episode + 1) % log_interval == 0:
                stats = self.metrics.get_current_stats()
                stats["epsilon_prisoner"] = self.agents["prisoner"].epsilon
                stats["epsilon_guard"] = self.agents["guard"].epsilon
                
                print(f"Épisode {episode + 1}/{num_episodes}")
                print(f"  Récompense prisoner: {stats['avg_reward_prisoner']:.2f}")
                print(f"  Récompense guard: {stats['avg_reward_guard']:.2f}")
                print(f"  Longueur moyenne: {stats['avg_episode_length']:.1f}")
                print(f"  Taux victoire prisoner: {stats['prisoner_win_rate']:.2%}")
                print(f"  Taux victoire guard: {stats['guard_win_rate']:.2%}")
                print(f"  Epsilon prisoner: {stats['epsilon_prisoner']:.3f}")
                print(f"  Epsilon guard: {stats['epsilon_guard']:.3f}")
                print(f"  Temps: {episode_time:.2f}s")
                print("-" * 50)
            
            # Sauvegarde périodique
            if (episode + 1) % save_interval == 0:
                self.save_models(episode + 1)
        
        # Sauvegarde finale
        self.save_models(num_episodes)
        
        if self.env_wrapper:
            self.env_wrapper.close()
        
        print("=== ENTRAÎNEMENT TERMINÉ ===")
    
    def save_models(self, episode):
        """Sauvegarde les modèles et métriques"""
        for agent_name, agent in self.agents.items():
            model_path = self.save_dir / f"{agent_name}_simple_episode_{episode}.json"
            agent.save(model_path)
        
        # Sauvegarde des métriques
        metrics_path = self.save_dir / f"metrics_episode_{episode}.json"
        self.metrics.save(metrics_path)
        
        print(f"Modèles sauvegardés pour l'épisode {episode}")

async def main():
    """Fonction principale"""
    print("=== ENTRAÎNEMENT PRISON ESCAPE SIMPLE ===")
    
    # Configuration
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    save_dir = "./simple_trained_models"
    
    print(f"Chemin expérience: {exp_path}")
    print(f"Répertoire sauvegarde: {save_dir}")
    
    # Création du trainer
    trainer = SimplePrisonEscapeTrainer(exp_path, save_dir)
    
    # Paramètres d'entraînement
    num_episodes = 300
    save_interval = 50
    log_interval = 10
    
    print(f"Paramètres: {num_episodes} épisodes, sauvegarde tous les {save_interval}")
    print("=" * 60)
    
    try:
        # Lancement de l'entraînement
        trainer.train(
            num_episodes=num_episodes,
            save_interval=save_interval,
            log_interval=log_interval
        )
        print("Entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        if trainer.env_wrapper:
            trainer.env_wrapper.close()

if __name__ == "__main__":
    asyncio.run(main())
