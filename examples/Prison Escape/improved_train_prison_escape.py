"""
Version améliorée de l'entraînement avec des mécanismes pour éviter la convergence
vers des stratégies défensives statiques
"""

import asyncio
import numpy as np
from pathlib import Path
import time
import json
from collections import deque

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

class ImprovedSimpleAgent:
    """Agent Q-Learning amélioré avec mécanismes anti-statiques"""
    
    def __init__(self, agent_name, grid_size=7):
        self.agent_name = agent_name
        self.grid_size = grid_size
        self.action_count = 4
        
        # Table Q avec valeurs par défaut optimistes
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.3  # Plus d'exploration
        self.epsilon_decay = 0.9995  # Décroissance plus lente
        self.epsilon_min = 0.05  # Minimum plus élevé
        self.gamma = 0.95
        
        # Mécanismes anti-statiques
        self.action_history = deque(maxlen=10)  # Historique des dernières actions
        self.position_history = deque(maxlen=20)  # Historique des positions
        self.static_penalty = 0.01  # Pénalité pour rester statique
        
        self.last_state = None
        self.last_action = None
        self.step_count = 0
        
    def get_state_key(self, observation):
        """Convertit l'observation en clé d'état"""
        return tuple(observation)
    
    def is_static_behavior(self):
        """Détecte si l'agent a un comportement statique"""
        if len(self.action_history) < 5:
            return False
        
        # Vérifie si les 5 dernières actions sont identiques
        recent_actions = list(self.action_history)[-5:]
        if len(set(recent_actions)) == 1:
            return True
        
        # Vérifie si l'agent reste dans la même zone
        if len(self.position_history) >= 10:
            positions = list(self.position_history)[-10:]
            if len(set(positions)) <= 2:  # Seulement 1-2 positions différentes
                return True
        
        return False
    
    def get_action(self, observation, training=True):
        """Choisit une action avec mécanismes anti-statiques"""
        state_key = self.get_state_key(observation)
        
        # Enregistrement de la position actuelle
        prisoner_pos = observation[0]
        self.position_history.append(prisoner_pos)
        
        # Initialisation de l'état si nouveau
        if state_key not in self.q_table:
            # Initialisation optimiste pour encourager l'exploration
            self.q_table[state_key] = np.random.uniform(0.1, 0.3, self.action_count)
        
        # Détection de comportement statique
        static_detected = self.is_static_behavior()
        
        # Calcul de l'epsilon adaptatif
        effective_epsilon = self.epsilon
        if static_detected:
            effective_epsilon = min(0.5, self.epsilon * 2)  # Force plus d'exploration
        
        if training and np.random.random() < effective_epsilon:
            # Exploration : évite de répéter la même action si statique
            if static_detected and len(self.action_history) > 0:
                last_action = self.action_history[-1]
                # Favorise une action différente
                available_actions = [a for a in range(self.action_count) if a != last_action]
                if available_actions:
                    action = np.random.choice(available_actions)
                else:
                    action = np.random.randint(0, self.action_count)
            else:
                action = np.random.randint(0, self.action_count)
        else:
            # Exploitation avec bruit anti-statique
            q_values = self.q_table[state_key].copy()
            
            # Pénalise légèrement la dernière action si comportement statique
            if static_detected and len(self.action_history) > 0:
                last_action = self.action_history[-1]
                q_values[last_action] -= 0.05
            
            action = np.argmax(q_values)
        
        self.action_history.append(action)
        self.last_state = state_key
        self.last_action = action
        self.step_count += 1
        
        return action
    
    def update(self, reward, next_observation, done):
        """Met à jour la table Q avec pénalités anti-statiques"""
        if self.last_state is None or self.last_action is None:
            return
        
        next_state_key = self.get_state_key(next_observation)
        
        # Initialisation du nouvel état si nécessaire
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.random.uniform(0.1, 0.3, self.action_count)
        
        # Calcul de la récompense ajustée
        adjusted_reward = reward
        
        # Pénalité pour comportement statique
        if self.is_static_behavior():
            adjusted_reward -= self.static_penalty
            
        # Bonus pour l'exploration (récompense intrinsèque)
        if len(set(self.action_history)) > 2:  # Diversité d'actions
            adjusted_reward += 0.005
        
        # Calcul de la valeur Q mise à jour
        if done:
            target = adjusted_reward
        else:
            target = adjusted_reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Mise à jour avec taux d'apprentissage adaptatif
        current_q = self.q_table[self.last_state][self.last_action]
        learning_rate = self.learning_rate
        if self.is_static_behavior():
            learning_rate *= 1.5  # Apprentissage plus rapide si statique
            
        self.q_table[self.last_state][self.last_action] = current_q + learning_rate * (target - current_q)
        
        # Décroissance de l'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset_episode(self):
        """Remet à zéro les variables d'épisode"""
        self.action_history.clear()
        self.position_history.clear()
        self.last_state = None
        self.last_action = None
        self.step_count = 0
    
    def save(self, filepath):
        """Sauvegarde l'agent"""
        data = {
            'agent_name': self.agent_name,
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'step_count': self.step_count
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
        self.step_count = data.get('step_count', 0)
        print(f"Agent {self.agent_name} chargé: {filepath}")

class ImprovedTrainer:
    """Trainer amélioré avec mécanismes de curriculum et de diversité"""
    
    def __init__(self, experiment_path, save_dir="./improved_models"):
        self.experiment_path = experiment_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.env_wrapper = None
        self.agents = {
            "prisoner": ImprovedSimpleAgent("prisoner"),
            "guard": ImprovedSimpleAgent("guard")
        }
        
        # Curriculum learning
        self.episode_count = 0
        self.diversity_bonus_episodes = []  # Épisodes avec bonus de diversité
        
        print(f"Trainer amélioré initialisé. Répertoire: {self.save_dir}")
    
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
    
    def calculate_diversity_bonus(self, episode_length, winner):
        """Calcule un bonus basé sur la diversité du jeu"""
        bonus = 0
        
        # Bonus pour les épisodes de longueur intéressante (ni trop courts, ni trop longs)
        if 20 <= episode_length <= 80:
            bonus += 0.1
        
        # Bonus si quelqu'un gagne (évite les matches nuls)
        if winner is not None:
            bonus += 0.2
        
        return bonus
    
    def run_episode(self, max_steps=200, training=True):
        """Exécute un épisode avec curriculum learning"""
        env = self.create_env()
        obs, infos = env.reset()
        
        # Reset des agents en début d'épisode
        for agent in self.agents.values():
            agent.reset_episode()
        
        total_rewards = {"prisoner": 0, "guard": 0}
        step_count = 0
        done = False
        winner = None
        prev_obs = {}
        
        # Curriculum learning : ajustement de max_steps
        if self.episode_count < 100:
            max_steps = min(150, max_steps)  # Épisodes plus courts au début
        
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
        
        # Bonus de diversité en fin d'épisode
        if training:
            diversity_bonus = self.calculate_diversity_bonus(step_count, winner)
            if diversity_bonus > 0:
                for agent_name in self.agents.keys():
                    total_rewards[agent_name] += diversity_bonus
                self.diversity_bonus_episodes.append(self.episode_count)
        
        self.episode_count += 1
        return total_rewards, step_count, winner
    
    def train(self, num_episodes=500, save_interval=50, log_interval=10):
        """Entraîne les agents avec curriculum learning"""
        print(f"=== ENTRAÎNEMENT AMÉLIORÉ ({num_episodes} épisodes) ===")
        
        metrics = {
            "episodes": [],
            "rewards": {"prisoner": [], "guard": []},
            "lengths": [],
            "winners": {"prisoner": 0, "guard": 0, "draw": 0},
            "diversity_episodes": []
        }
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Exécution d'un épisode d'entraînement
            total_rewards, episode_length, winner = self.run_episode(training=True)
            
            # Mise à jour des métriques
            metrics["episodes"].append(episode)
            metrics["rewards"]["prisoner"].append(total_rewards["prisoner"])
            metrics["rewards"]["guard"].append(total_rewards["guard"])
            metrics["lengths"].append(episode_length)
            
            if winner:
                metrics["winners"][winner] += 1
            else:
                metrics["winners"]["draw"] += 1
            
            episode_time = time.time() - start_time
            
            # Logging périodique
            if (episode + 1) % log_interval == 0:
                recent_rewards_p = metrics["rewards"]["prisoner"][-log_interval:]
                recent_rewards_g = metrics["rewards"]["guard"][-log_interval:]
                recent_lengths = metrics["lengths"][-log_interval:]
                
                print(f"Épisode {episode + 1}/{num_episodes}")
                print(f"  Récompense prisoner: {np.mean(recent_rewards_p):.2f}")
                print(f"  Récompense guard: {np.mean(recent_rewards_g):.2f}")
                print(f"  Longueur moyenne: {np.mean(recent_lengths):.1f}")
                
                # Calcul des taux de victoire récents
                recent_winners = {"prisoner": 0, "guard": 0, "draw": 0}
                for i in range(max(0, len(metrics["lengths"]) - log_interval), len(metrics["lengths"])):
                    ep_rewards = {"prisoner": metrics["rewards"]["prisoner"][i], 
                                "guard": metrics["rewards"]["guard"][i]}
                    if ep_rewards["prisoner"] > ep_rewards["guard"]:
                        recent_winners["prisoner"] += 1
                    elif ep_rewards["guard"] > ep_rewards["prisoner"]:
                        recent_winners["guard"] += 1
                    else:
                        recent_winners["draw"] += 1
                
                p_rate = recent_winners["prisoner"] / log_interval * 100
                g_rate = recent_winners["guard"] / log_interval * 100
                d_rate = recent_winners["draw"] / log_interval * 100
                
                print(f"  Victoires récentes - P: {p_rate:.0f}%, G: {g_rate:.0f}%, D: {d_rate:.0f}%")
                print(f"  Epsilon - P: {self.agents['prisoner'].epsilon:.3f}, G: {self.agents['guard'].epsilon:.3f}")
                print(f"  Épisodes bonus: {len(self.diversity_bonus_episodes)}")
                print(f"  Temps: {episode_time:.2f}s")
                print("-" * 50)
            
            # Sauvegarde périodique
            if (episode + 1) % save_interval == 0:
                self.save_models(episode + 1)
                self.save_metrics(metrics, episode + 1)
        
        # Sauvegarde finale
        self.save_models(num_episodes)
        self.save_metrics(metrics, num_episodes)
        
        if self.env_wrapper:
            self.env_wrapper.close()
        
        print("=== ENTRAÎNEMENT AMÉLIORÉ TERMINÉ ===")
        self.print_final_stats(metrics)
    
    def save_models(self, episode):
        """Sauvegarde les modèles"""
        for agent_name, agent in self.agents.items():
            model_path = self.save_dir / f"{agent_name}_improved_episode_{episode}.json"
            agent.save(model_path)
        print(f"Modèles améliorés sauvegardés pour l'épisode {episode}")
    
    def save_metrics(self, metrics, episode):
        """Sauvegarde les métriques"""
        metrics_path = self.save_dir / f"improved_metrics_episode_{episode}.json"
        # Conversion pour JSON
        json_metrics = {
            "episodes": metrics["episodes"],
            "rewards": metrics["rewards"],
            "lengths": metrics["lengths"],
            "winners": metrics["winners"],
            "diversity_episodes": self.diversity_bonus_episodes
        }
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
    
    def print_final_stats(self, metrics):
        """Affiche les statistiques finales"""
        total_episodes = len(metrics["episodes"])
        
        print(f"\n=== STATISTIQUES FINALES ({total_episodes} épisodes) ===")
        print(f"Récompense finale prisoner: {np.mean(metrics['rewards']['prisoner'][-50:]):.2f}")
        print(f"Récompense finale guard: {np.mean(metrics['rewards']['guard'][-50:]):.2f}")
        print(f"Longueur finale moyenne: {np.mean(metrics['lengths'][-50:]):.1f}")
        
        print(f"\nVictoires totales:")
        for agent, count in metrics["winners"].items():
            percentage = count / total_episodes * 100
            print(f"  {agent.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\nÉpisodes avec bonus de diversité: {len(self.diversity_bonus_episodes)} ({len(self.diversity_bonus_episodes)/total_episodes*100:.1f}%)")

async def main():
    """Fonction principale pour l'entraînement amélioré"""
    print("=== ENTRAÎNEMENT PRISON ESCAPE AMÉLIORÉ ===")
    
    # Configuration
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    save_dir = "./improved_trained_models"
    
    print(f"Chemin expérience: {exp_path}")
    print(f"Répertoire sauvegarde: {save_dir}")
    
    # Création du trainer
    trainer = ImprovedTrainer(exp_path, save_dir)
    
    # Paramètres d'entraînement
    num_episodes = 400
    save_interval = 50
    log_interval = 10
    
    print(f"Paramètres: {num_episodes} épisodes, sauvegarde tous les {save_interval}")
    print("Améliorations:")
    print("- Pénalités anti-statiques")
    print("- Bonus de diversité")
    print("- Curriculum learning")
    print("- Exploration adaptative")
    print("=" * 60)
    
    try:
        # Lancement de l'entraînement
        trainer.train(
            num_episodes=num_episodes,
            save_interval=save_interval,
            log_interval=log_interval
        )
        print("Entraînement amélioré terminé avec succès!")
        
    except Exception as e:
        print(f"Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        if trainer.env_wrapper:
            trainer.env_wrapper.close()

if __name__ == "__main__":
    asyncio.run(main())
