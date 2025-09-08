"""
Script d'évaluation des agents entraînés dans Prison Escape
Permet de tester les performances des modèles entraînés et de visualiser leur comportement
"""

import asyncio
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pickle

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv
from stable_baselines3 import PPO

class PrisonEscapeEvaluator:
    """Classe pour évaluer les agents entraînés"""
    
    def __init__(self, experiment_path, models_dir="./trained_models"):
        self.experiment_path = experiment_path
        self.models_dir = Path(models_dir)
        self.models = {}
        self.env = None
    
    def load_models(self, episode_number=None):
        """Charge les modèles entraînés"""
        if episode_number is None:
            # Trouve le dernier épisode disponible
            prisoner_files = list(self.models_dir.glob("prisoner_ppo_episode_*.zip"))
            if prisoner_files:
                episode_numbers = [int(f.stem.split('_')[-1]) for f in prisoner_files]
                episode_number = max(episode_numbers)
            else:
                raise FileNotFoundError("Aucun modèle trouvé dans le répertoire")
        
        for agent in ["prisoner", "guard"]:
            model_path = self.models_dir / f"{agent}_ppo_episode_{episode_number}.zip"
            if model_path.exists():
                self.models[agent] = PPO.load(str(model_path))
                print(f"Modèle {agent} chargé depuis {model_path}")
            else:
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        return episode_number
    
    async def create_env(self):
        """Crée l'environnement de test"""
        if self.env is None:
            self.env = GamaParallelEnv(
                gaml_experiment_path=self.experiment_path,
                gaml_experiment_name="main",
                gama_ip_address="localhost",
                gama_port=1001
            )
        return self.env
    
    async def run_evaluation_episode(self, max_steps=200, render=False):
        """Exécute un épisode d'évaluation"""
        env = await self.create_env()
        obs, infos = env.reset()
        
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "positions": {"prisoner": [], "guard": [], "escape": []}
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
                    action, _ = self.models[agent].predict(obs_array, deterministic=True)
                    actions[agent] = int(action)
                    
                    # Extraction des positions pour visualisation
                    prisoner_pos = obs_array[0]
                    guard_pos = obs_array[1]
                    escape_pos = obs_array[2]
                    
                    if agent == "prisoner":  # Stockage une seule fois par étape
                        episode_data["positions"]["prisoner"].append(prisoner_pos)
                        episode_data["positions"]["guard"].append(guard_pos)
                        episode_data["positions"]["escape"].append(escape_pos)
            
            episode_data["actions"].append(actions.copy())
            
            if render:
                self.print_game_state(obs, actions, step_count)
            
            # Exécution de l'étape
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            episode_data["rewards"].append(rewards.copy())
            
            # Mise à jour des récompenses totales
            for agent in ["prisoner", "guard"]:
                if agent in rewards:
                    reward = rewards[agent]
                    total_rewards[agent] += reward
                    
                    # Détermination du gagnant
                    if terminations.get(agent, False) and reward > 0:
                        winner = agent
            
            obs = next_obs
            step_count += 1
            
            # Vérification de fin d'épisode
            if any(terminations.values()) or any(truncations.values()):
                done = True
                if render:
                    print(f"\n=== FIN DE L'ÉPISODE ===")
                    print(f"Gagnant: {winner if winner else 'Match nul'}")
                    print(f"Récompenses finales: {total_rewards}")
                    print(f"Nombre d'étapes: {step_count}")
        
        return episode_data, total_rewards, step_count, winner
    
    def print_game_state(self, obs, actions, step):
        """Affiche l'état du jeu de manière lisible"""
        if "prisoner" in obs:
            obs_array = np.array(obs["prisoner"])
            prisoner_pos = obs_array[0]
            guard_pos = obs_array[1]
            escape_pos = obs_array[2]
            
            # Conversion en coordonnées x, y (grille 7x7)
            prisoner_x, prisoner_y = prisoner_pos % 7, prisoner_pos // 7
            guard_x, guard_y = guard_pos % 7, guard_pos // 7
            escape_x, escape_y = escape_pos % 7, escape_pos // 7
            
            print(f"\n--- Étape {step} ---")
            print(f"Prisoner: ({prisoner_x}, {prisoner_y}) -> Action: {actions.get('prisoner', 'N/A')}")
            print(f"Guard: ({guard_x}, {guard_y}) -> Action: {actions.get('guard', 'N/A')}")
            print(f"Escape: ({escape_x}, {escape_y})")
    
    async def evaluate(self, num_episodes=100, render_episodes=5):
        """Évalue les modèles sur plusieurs épisodes"""
        print("=== ÉVALUATION DES AGENTS ===")
        
        results = {
            "episodes": [],
            "total_rewards": {"prisoner": [], "guard": []},
            "episode_lengths": [],
            "winners": {"prisoner": 0, "guard": 0, "draw": 0}
        }
        
        for episode in range(num_episodes):
            render = episode < render_episodes
            
            if render:
                print(f"\n=== ÉPISODE {episode + 1} (avec affichage) ===")
            
            episode_data, total_rewards, episode_length, winner = await self.run_evaluation_episode(render=render)
            
            # Stockage des résultats
            results["episodes"].append(episode_data)
            results["total_rewards"]["prisoner"].append(total_rewards["prisoner"])
            results["total_rewards"]["guard"].append(total_rewards["guard"])
            results["episode_lengths"].append(episode_length)
            
            if winner:
                results["winners"][winner] += 1
            else:
                results["winners"]["draw"] += 1
            
            if (episode + 1) % 10 == 0:
                print(f"Épisode {episode + 1}/{num_episodes} terminé")
        
        # Calcul des statistiques finales
        self.print_evaluation_results(results)
        return results
    
    def print_evaluation_results(self, results):
        """Affiche les résultats de l'évaluation"""
        num_episodes = len(results["episode_lengths"])
        
        print(f"\n=== RÉSULTATS D'ÉVALUATION ({num_episodes} épisodes) ===")
        print(f"Récompense moyenne prisoner: {np.mean(results['total_rewards']['prisoner']):.2f} ± {np.std(results['total_rewards']['prisoner']):.2f}")
        print(f"Récompense moyenne guard: {np.mean(results['total_rewards']['guard']):.2f} ± {np.std(results['total_rewards']['guard']):.2f}")
        print(f"Longueur moyenne des épisodes: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
        
        print(f"\nTaux de victoire:")
        print(f"  Prisoner: {results['winners']['prisoner']/num_episodes:.1%}")
        print(f"  Guard: {results['winners']['guard']/num_episodes:.1%}")
        print(f"  Match nul: {results['winners']['draw']/num_episodes:.1%}")
    
    def plot_evaluation_results(self, results, save_path="evaluation_results.png"):
        """Crée des graphiques des résultats d'évaluation"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution des récompenses
        axes[0, 0].hist(results["total_rewards"]["prisoner"], alpha=0.7, label="Prisoner", bins=20)
        axes[0, 0].hist(results["total_rewards"]["guard"], alpha=0.7, label="Guard", bins=20)
        axes[0, 0].set_title("Distribution des récompenses totales")
        axes[0, 0].set_xlabel("Récompense")
        axes[0, 0].set_ylabel("Fréquence")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Distribution des longueurs d'épisode
        axes[0, 1].hist(results["episode_lengths"], bins=20, alpha=0.7)
        axes[0, 1].set_title("Distribution des longueurs d'épisode")
        axes[0, 1].set_xlabel("Nombre d'étapes")
        axes[0, 1].set_ylabel("Fréquence")
        axes[0, 1].grid(True)
        
        # Évolution des récompenses
        axes[1, 0].plot(results["total_rewards"]["prisoner"], label="Prisoner", alpha=0.7)
        axes[1, 0].plot(results["total_rewards"]["guard"], label="Guard", alpha=0.7)
        axes[1, 0].set_title("Évolution des récompenses par épisode")
        axes[1, 0].set_xlabel("Épisode")
        axes[1, 0].set_ylabel("Récompense")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Taux de victoire
        winners = ["Prisoner", "Guard", "Match nul"]
        counts = [results["winners"]["prisoner"], results["winners"]["guard"], results["winners"]["draw"]]
        colors = ['orange', 'blue', 'gray']
        
        axes[1, 1].pie(counts, labels=winners, colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title("Répartition des victoires")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Graphiques sauvegardés: {save_path}")
    
    async def close(self):
        """Ferme l'environnement"""
        if self.env:
            self.env.close()

async def main():
    """Fonction principale pour l'évaluation"""
    
    # Configuration
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    models_dir = "./trained_models"
    
    # Création de l'évaluateur
    evaluator = PrisonEscapeEvaluator(exp_path, models_dir)
    
    try:
        # Chargement des modèles
        episode_number = evaluator.load_models()
        print(f"Modèles de l'épisode {episode_number} chargés avec succès")
        
        # Évaluation
        results = await evaluator.evaluate(num_episodes=50, render_episodes=3)
        
        # Visualisation des résultats
        evaluator.plot_evaluation_results(results)
        
    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
    
    finally:
        await evaluator.close()

if __name__ == "__main__":
    asyncio.run(main())
