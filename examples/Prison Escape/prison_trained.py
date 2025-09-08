import asyncio
import json
import time
import numpy as np
from pathlib import Path

import pettingzoo
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv


class SimpleTrainedAgent:
    """Agent simple avec modèle Q-Learning entraîné"""
    
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.q_table = {}
        self.epsilon = 0.15  # Exploration pour mode stochastique
        
    def get_state_key(self, observation):
        """Convertit l'observation en clé d'état"""
        return tuple(observation)
    
    def get_action_stochastic(self, observation):
        """Choisit une action en mode stochastique (avec exploration)"""
        state_key = self.get_state_key(observation)
        
        if state_key not in self.q_table:
            # État non vu, action aléatoire
            return np.random.randint(0, 4)
        
        # Mode stochastique avec epsilon-greedy
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.randint(0, 4)
        else:
            # Exploitation avec les Q-values apprises
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def load_model(self, filepath):
        """Charge le modèle depuis un fichier JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.agent_name = data['agent_name']
            self.q_table = {eval(k): np.array(v) for k, v in data['q_table'].items()}
            
            print(f"✅ Agent {self.agent_name} chargé avec {len(self.q_table)} états")
            return True
        except FileNotFoundError:
            print(f"❌ Modèle {filepath} non trouvé, utilisation d'actions aléatoires")
            return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {filepath}: {e}")
            return False


async def main():
    print("🎮 PRISON ESCAPE - Exécution avec agents entraînés (mode stochastique)")
    print("=" * 70)
    
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    exp_name = "main"

    env = GamaParallelEnv(gaml_experiment_path=exp_path,
                          gaml_experiment_name=exp_name,
                          gama_ip_address="localhost",
                          gama_port=1000)
    
    # Création et chargement des agents entraînés
    agents = {}
    models_dir = Path("./improved_trained_models")
    
    for agent_name in ["prisoner", "guard"]:
        agent = SimpleTrainedAgent(agent_name)
        
        # Tentative de chargement du modèle le plus récent
        model_files = list(models_dir.glob(f"{agent_name}_improved_episode_*.json"))
        if model_files:
            # Trouve le modèle du dernier épisode
            episode_numbers = [int(f.stem.split('_')[-1]) for f in model_files]
            latest_episode = max(episode_numbers)
            model_path = models_dir / f"{agent_name}_improved_episode_{latest_episode}.json"
            agent.load_model(model_path)
        else:
            print(f"⚠️ Aucun modèle trouvé pour {agent_name}, utilisation d'actions aléatoires")
        
        agents[agent_name] = agent
    
    # Initialisation de la partie
    obs, infos = env.reset()
    print(f"\n🎯 Observations initiales:")
    for agent_name, observation in obs.items():
        print(f"   {agent_name}: {observation}")
    
    step_count = 0
    total_rewards = {"prisoner": 0, "guard": 0}
    
    print(f"\n🚀 Début de la partie (mode stochastique, ε={agents['prisoner'].epsilon})...")
    print("-" * 70)
    
    time.sleep(5)
    
    # while env.agents and step_count < 200:  # Limite à 200 étapes
    for step in range(300):
        # step_count += 1
        actions = {}
        
        # Choix des actions par les agents entraînés
        for agent_name in env.agents:
            if agent_name in obs:
                observation = np.array(obs[agent_name])
                action = agents[agent_name].get_action_stochastic(observation)
                actions[agent_name] = int(action)
        
        # Exécution de l'étape
        obs, rewards, terminations, truncations, info = env.step(actions)
        time.sleep(0.2)
        
        # Mise à jour des récompenses totales
        for agent_name in ["prisoner", "guard"]:
            if agent_name in rewards:
                total_rewards[agent_name] += rewards[agent_name]
        
        # Affichage de l'état
        action_names = ["⬅️", "➡️", "⬆️", "⬇️"]
        actions_str = {k: action_names[v] for k, v in actions.items()}
        
        print(f"Tour {step_count:3d} | Actions: {actions_str} | "
              f"Récompenses: {rewards} | "
              f"Fin: {any(terminations.values()) or any(truncations.values())}")
        
        # Vérification de fin de partie
        if any(terminations.values()) or any(truncations.values()):
            obs, infos = env.reset()
            time.sleep(5)
    
    print("-" * 70)
    print(f"🏁 Fin de partie après {step_count} tours")
    print(f"📊 Récompenses totales: {total_rewards}")
    
    # Analyse du résultat
    if any(terminations.values()):
        for agent_name, terminated in terminations.items():
            if terminated and rewards.get(agent_name, 0) > 0:
                if agent_name == "prisoner":
                    print("🎉 Victoire du PRISONER! Il s'est échappé!")
                else:
                    print("🎉 Victoire du GUARD! Il a capturé le prisoner!")
                break
        else:
            print("⚖️ Match nul - Aucun vainqueur clair")
    else:
        print("⏱️ Match nul - Limite de temps atteinte")
    
    # Statistiques finales
    if step_count < 50:
        print("⚡ Partie rapide - Confrontation directe")
    elif step_count < 150:
        print("🎯 Partie normale - Bon équilibre")
    else:
        print("🐌 Partie longue - Stratégies prudentes")

    env.close()
    
if __name__ == "__main__":
    asyncio.run(main())