"""
Script d'évaluation simple pour les agents entraînés avec Q-Learning
Teste les performances des modèles sauvegardés
"""

import asyncio
import numpy as np
import json
from pathlib import Path
import time

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

class SimpleAgentEvaluator:
    """Classe pour charger et évaluer les agents entraînés"""
    
    def __init__(self, agent_name, grid_size=7):
        self.agent_name = agent_name
        self.grid_size = grid_size
        self.action_count = 4
        self.q_table = {}
        
    def get_state_key(self, observation):
        """Convertit l'observation en clé d'état"""
        return tuple(observation)
    
    def get_action(self, observation, deterministic=True):
        """Choisit une action basée sur l'observation"""
        state_key = self.get_state_key(observation)
        
        if state_key not in self.q_table:
            # État non vu, action aléatoire
            return np.random.randint(0, self.action_count)
        
        if deterministic:
            # Mode déterministe : meilleure action
            return np.argmax(self.q_table[state_key])
        else:
            # Mode stochastique avec un peu d'exploration
            if np.random.random() < 0.1:  # 10% d'exploration
                return np.random.randint(0, self.action_count)
            else:
                return np.argmax(self.q_table[state_key])
    
    def load(self, filepath):
        """Charge l'agent depuis un fichier JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.agent_name = data['agent_name']
        self.q_table = {eval(k): np.array(v) for k, v in data['q_table'].items()}
        print(f"Agent {self.agent_name} chargé depuis {filepath}")
        print(f"  États appris: {len(self.q_table)}")

class SimpleEvaluator:
    """Évaluateur pour les agents entraînés"""
    
    def __init__(self, experiment_path, models_dir="./simple_trained_models"):
        self.experiment_path = experiment_path
        self.models_dir = Path(models_dir)
        self.agents = {}
        self.env = None
    
    def load_agents(self, episode_number=None):
        """Charge les agents d'un épisode spécifique"""
        if episode_number is None:
            # Trouve le dernier épisode disponible
            prisoner_files = list(self.models_dir.glob("prisoner_simple_episode_*.json"))
            if prisoner_files:
                episode_numbers = [int(f.stem.split('_')[-1]) for f in prisoner_files]
                episode_number = max(episode_numbers)
            else:
                raise FileNotFoundError("Aucun modèle trouvé")
        
        # Chargement des agents
        for agent_name in ["prisoner", "guard"]:
            model_path = self.models_dir / f"{agent_name}_simple_episode_{episode_number}.json"
            if model_path.exists():
                agent = SimpleAgentEvaluator(agent_name)
                agent.load(model_path)
                self.agents[agent_name] = agent
            else:
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        return episode_number
    
    def create_env(self):
        """Crée l'environnement GAMA"""
        if self.env is None:
            self.env = GamaParallelEnv(
                gaml_experiment_path=self.experiment_path,
                gaml_experiment_name="main",
                gama_ip_address="localhost",
                gama_port=1001
            )
        return self.env
    
    def run_evaluation_episode(self, max_steps=200, verbose=False):
        """Exécute un épisode d'évaluation"""
        env = self.create_env()
        obs, infos = env.reset()
        
        total_rewards = {"prisoner": 0, "guard": 0}
        step_count = 0
        done = False
        winner = None
        
        episode_log = []
        
        while not done and step_count < max_steps:
            actions = {}
            step_info = {"step": step_count + 1}
            
            # Collecte des actions
            for agent_name in ["prisoner", "guard"]:
                if agent_name in obs:
                    obs_array = np.array(obs[agent_name])
                    action = self.agents[agent_name].get_action(obs_array, deterministic=True)
                    actions[agent_name] = int(action)
                    
                    # Ajout d'infos pour le log
                    step_info[f"{agent_name}_obs"] = obs_array.tolist()
                    step_info[f"{agent_name}_action"] = action
            
            if verbose:
                self.print_step_info(step_count + 1, obs, actions)
            
            # Exécution de l'étape
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            step_info["rewards"] = rewards
            step_info["terminations"] = terminations
            episode_log.append(step_info)
            
            # Mise à jour des récompenses
            for agent_name in ["prisoner", "guard"]:
                if agent_name in rewards:
                    reward = rewards[agent_name]
                    total_rewards[agent_name] += reward
                    
                    if terminations.get(agent_name, False) and reward > 0:
                        winner = agent_name
            
            obs = next_obs
            step_count += 1
            
            if any(terminations.values()) or any(truncations.values()):
                done = True
        
        return total_rewards, step_count, winner, episode_log
    
    def print_step_info(self, step, obs, actions):
        """Affiche les informations d'une étape"""
        if "prisoner" in obs:
            p_pos = obs["prisoner"][0]
            g_pos = obs["prisoner"][1]
            e_pos = obs["prisoner"][2]
            
            # Conversion en coordonnées (x, y)
            p_x, p_y = p_pos % 7, p_pos // 7
            g_x, g_y = g_pos % 7, g_pos // 7
            e_x, e_y = e_pos % 7, e_pos // 7
            
            action_names = ["Gauche", "Droite", "Haut", "Bas"]
            
            print(f"Étape {step}:")
            print(f"  Prisoner: ({p_x}, {p_y}) -> {action_names[actions.get('prisoner', 0)]}")
            print(f"  Guard: ({g_x}, {g_y}) -> {action_names[actions.get('guard', 0)]}")
            print(f"  Escape: ({e_x}, {e_y})")
    
    def evaluate(self, num_episodes=50, verbose_episodes=3):
        """Évalue les agents sur plusieurs épisodes"""
        print("=== ÉVALUATION DES AGENTS ENTRAÎNÉS ===")
        
        results = {
            "total_rewards": {"prisoner": [], "guard": []},
            "episode_lengths": [],
            "winners": {"prisoner": 0, "guard": 0, "draw": 0},
            "detailed_logs": []
        }
        
        for episode in range(num_episodes):
            verbose = episode < verbose_episodes
            
            if verbose:
                print(f"\n=== ÉPISODE {episode + 1} (détaillé) ===")
            
            total_rewards, episode_length, winner, episode_log = self.run_evaluation_episode(verbose=verbose)
            
            # Stockage des résultats
            results["total_rewards"]["prisoner"].append(total_rewards["prisoner"])
            results["total_rewards"]["guard"].append(total_rewards["guard"])
            results["episode_lengths"].append(episode_length)
            
            if winner:
                results["winners"][winner] += 1
            else:
                results["winners"]["draw"] += 1
            
            if verbose:
                print(f"Résultat: Gagnant={winner or 'Match nul'}, Étapes={episode_length}, Récompenses={total_rewards}")
            
            if (episode + 1) % 10 == 0:
                print(f"Épisode {episode + 1}/{num_episodes} terminé")
        
        self.print_results(results, num_episodes)
        return results
    
    def print_results(self, results, num_episodes):
        """Affiche les résultats de l'évaluation"""
        print(f"\n=== RÉSULTATS ({num_episodes} épisodes) ===")
        
        p_rewards = results["total_rewards"]["prisoner"]
        g_rewards = results["total_rewards"]["guard"]
        lengths = results["episode_lengths"]
        
        print(f"Récompense moyenne prisoner: {np.mean(p_rewards):.2f} ± {np.std(p_rewards):.2f}")
        print(f"Récompense moyenne guard: {np.mean(g_rewards):.2f} ± {np.std(g_rewards):.2f}")
        print(f"Longueur moyenne: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        
        print(f"\nRépartition des victoires:")
        for agent, count in results["winners"].items():
            percentage = count / num_episodes * 100
            print(f"  {agent.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Analyse de la convergence
        if all(r == 0 for r in p_rewards + g_rewards):
            print(f"\n🤔 Observation: Tous les épisodes se terminent par timeout")
            print("   Les agents ont appris à éviter les confrontations directes")
        
        if np.mean(lengths) > 150:
            print(f"\n⏱️  Les épisodes sont longs (moy: {np.mean(lengths):.1f} étapes)")
            print("   Cela suggère que les agents jouent de manière défensive")
    
    def close(self):
        """Ferme l'environnement"""
        if self.env:
            self.env.close()

def list_available_models(models_dir="./simple_trained_models"):
    """Liste les modèles disponibles"""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Répertoire non trouvé: {models_dir}")
        return []
    
    prisoner_files = list(models_path.glob("prisoner_simple_episode_*.json"))
    if not prisoner_files:
        print("Aucun modèle trouvé")
        return []
    
    episodes = sorted([int(f.stem.split('_')[-1]) for f in prisoner_files])
    print(f"Modèles disponibles: épisodes {episodes}")
    return episodes

async def main():
    """Fonction principale"""
    print("=== ÉVALUATEUR D'AGENTS PRISON ESCAPE ===")
    
    # Configuration
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    models_dir = "./simple_trained_models"
    
    # Liste des modèles disponibles
    available_episodes = list_available_models(models_dir)
    if not available_episodes:
        return
    
    # Création de l'évaluateur
    evaluator = SimpleEvaluator(exp_path, models_dir)
    
    try:
        # Chargement du dernier modèle
        episode_number = evaluator.load_agents()
        print(f"\nModèles de l'épisode {episode_number} chargés")
        
        # Évaluation
        print(f"\nDébut de l'évaluation...")
        results = evaluator.evaluate(num_episodes=30, verbose_episodes=2)
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        evaluator.close()

if __name__ == "__main__":
    asyncio.run(main())
