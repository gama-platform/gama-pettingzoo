"""
Script pour exécuter une partie en temps réel avec les agents améliorés entraînés
Affiche la grille, les mouvements et l'analyse stratégique en direct
"""

import asyncio
import numpy as np
import json
import time
from pathlib import Path
from collections import deque

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

class ImprovedGameAgent:
    """Agent amélioré pour jouer une partie"""
    
    def __init__(self, agent_name, grid_size=7):
        self.agent_name = agent_name
        self.grid_size = grid_size
        self.action_count = 4
        self.q_table = {}
        
        # Historique pour l'analyse
        self.move_history = deque(maxlen=10)
        self.decision_confidence = 0.0
        
    def get_state_key(self, observation):
        """Convertit l'observation en clé d'état"""
        return tuple(observation)
    
    def get_action_with_analysis(self, observation, deterministic=True):
        """Choisit une action avec analyse de la décision"""
        state_key = self.get_state_key(observation)
        
        if state_key not in self.q_table:
            # État non vu, action aléatoire
            action = np.random.randint(0, self.action_count)
            self.decision_confidence = 0.25  # Faible confiance
            decision_type = "EXPLORATION (état inconnu)"
        else:
            q_values = self.q_table[state_key]
            
            if deterministic:
                # Mode déterministe
                action = np.argmax(q_values)
                max_q = np.max(q_values)
                second_max_q = np.partition(q_values, -2)[-2]
                
                # Calcul de la confiance basé sur l'écart entre les meilleures actions
                if max_q > second_max_q:
                    self.decision_confidence = min(1.0, (max_q - second_max_q) / abs(max_q) if max_q != 0 else 1.0)
                else:
                    self.decision_confidence = 0.5
                
                decision_type = "EXPLOITATION"
            else:
                # Mode avec exploration (10%)
                if np.random.random() < 0.1:
                    action = np.random.randint(0, self.action_count)
                    self.decision_confidence = 0.1
                    decision_type = "EXPLORATION (aléatoire)"
                else:
                    action = np.argmax(q_values)
                    max_q = np.max(q_values)
                    second_max_q = np.partition(q_values, -2)[-2]
                    self.decision_confidence = min(1.0, (max_q - second_max_q) / abs(max_q) if max_q != 0 else 1.0)
                    decision_type = "EXPLOITATION"
        
        # Enregistrement du mouvement
        self.move_history.append({
            "action": action,
            "confidence": self.decision_confidence,
            "type": decision_type
        })
        
        return action, decision_type, self.decision_confidence
    
    def get_strategy_analysis(self, observation):
        """Analyse la stratégie actuelle de l'agent"""
        p_pos = observation[0]
        g_pos = observation[1]
        e_pos = observation[2]
        
        # Conversion en coordonnées
        p_x, p_y = p_pos % 7, p_pos // 7
        g_x, g_y = g_pos % 7, g_pos // 7
        e_x, e_y = e_pos % 7, e_pos // 7
        
        # Calcul des distances
        if self.agent_name == "prisoner":
            dist_to_goal = abs(p_x - e_x) + abs(p_y - e_y)
            dist_to_threat = abs(p_x - g_x) + abs(p_y - g_y)
            
            if dist_to_threat <= 2:
                strategy = "🏃 FUITE (garde proche)"
            elif dist_to_goal <= 3:
                strategy = "🎯 APPROCHE FINALE"
            elif dist_to_threat > 5:
                strategy = "➡️ PROGRESSION SÛRE"
            else:
                strategy = "⚖️ ÉQUILIBRAGE"
                
        else:  # guard
            dist_to_target = abs(g_x - p_x) + abs(g_y - p_y)
            dist_prisoner_to_escape = abs(p_x - e_x) + abs(p_y - e_y)
            
            if dist_to_target <= 2:
                strategy = "⚡ CAPTURE IMMINENTE"
            elif dist_prisoner_to_escape <= 3:
                strategy = "🚫 INTERCEPTION"
            elif dist_to_target <= 4:
                strategy = "🎯 POURSUITE ACTIVE"
            else:
                strategy = "🔍 RECHERCHE"
        
        return strategy
    
    def load(self, filepath):
        """Charge l'agent depuis un fichier JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.agent_name = data['agent_name']
        self.q_table = {eval(k): np.array(v) for k, v in data['q_table'].items()}
        
        print(f"✅ Agent {self.agent_name} chargé avec {len(self.q_table)} états appris")

class GameVisualizer:
    """Classe pour visualiser la partie en cours"""
    
    def __init__(self):
        self.action_names = ["⬅️ Gauche", "➡️ Droite", "⬆️ Haut", "⬇️ Bas"]
        self.move_count = 0
        
    def clear_screen(self):
        """Efface l'écran (compatible Windows/Linux)"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_grid(self, prisoner_pos, guard_pos, escape_pos):
        """Dessine la grille de jeu"""
        # Conversion en coordonnées
        p_x, p_y = prisoner_pos % 7, prisoner_pos // 7
        g_x, g_y = guard_pos % 7, guard_pos // 7
        e_x, e_y = escape_pos % 7, escape_pos // 7
        
        print("┌" + "─" * 21 + "┐")
        
        for y in range(7):
            row = "│"
            for x in range(7):
                if x == p_x and y == p_y:
                    if x == g_x and y == g_y:
                        cell = "💥"  # Collision
                    else:
                        cell = "🟠"  # Prisoner
                elif x == g_x and y == g_y:
                    cell = "🔵"  # Guard
                elif x == e_x and y == e_y:
                    cell = "🟫"  # Escape
                else:
                    cell = "⬜"  # Empty
                
                row += cell + " "
            row += "│"
            print(row)
        
        print("└" + "─" * 21 + "┘")
    
    def print_game_state(self, step, obs, actions, strategies, confidences, decision_types):
        """Affiche l'état complet du jeu"""
        self.clear_screen()
        self.move_count = step
        
        # En-tête
        print("=" * 60)
        print(f"🎮 PRISON ESCAPE - PARTIE EN DIRECT (Tour {step})")
        print("=" * 60)
        
        # Extraction des positions
        p_pos = obs["prisoner"][0]
        g_pos = obs["prisoner"][1]
        e_pos = obs["prisoner"][2]
        
        # Affichage de la grille
        print("\n🗺️ GRILLE DE JEU:")
        self.draw_grid(p_pos, g_pos, e_pos)
        
        # Conversion en coordonnées pour l'affichage
        p_x, p_y = p_pos % 7, p_pos // 7
        g_x, g_y = g_pos % 7, g_pos // 7
        e_x, e_y = e_pos % 7, e_pos // 7
        
        # Calcul des distances
        dist_pg = abs(p_x - g_x) + abs(p_y - g_y)
        dist_pe = abs(p_x - e_x) + abs(p_y - e_y)
        
        # Informations des agents
        print(f"\n📊 ÉTAT DES AGENTS:")
        print(f"┌─ 🟠 PRISONER ─────────────────────────────────────┐")
        print(f"│ Position: ({p_x}, {p_y}) | Distance évasion: {dist_pe}    │")
        print(f"│ Action: {self.action_names[actions.get('prisoner', 0)]:15} │")
        print(f"│ Stratégie: {strategies.get('prisoner', 'Inconnue'):25} │")
        print(f"│ Confiance: {'█' * int(confidences.get('prisoner', 0) * 10):10} {confidences.get('prisoner', 0):.2f} │")
        print(f"│ Type: {decision_types.get('prisoner', 'INCONNU'):30} │")
        print("└───────────────────────────────────────────────────┘")
        
        print(f"┌─ 🔵 GUARD ────────────────────────────────────────┐")
        print(f"│ Position: ({g_x}, {g_y}) | Distance prisoner: {dist_pg}   │")
        print(f"│ Action: {self.action_names[actions.get('guard', 0)]:15} │")
        print(f"│ Stratégie: {strategies.get('guard', 'Inconnue'):25} │")
        print(f"│ Confiance: {'█' * int(confidences.get('guard', 0) * 10):10} {confidences.get('guard', 0):.2f} │")
        print(f"│ Type: {decision_types.get('guard', 'INCONNU'):30} │")
        print("└───────────────────────────────────────────────────┘")
        
        # Analyse tactique
        print(f"\n🎯 ANALYSE TACTIQUE:")
        if dist_pg <= 1:
            print("🚨 CONTACT IMMINENT! Le garde peut capturer au prochain tour!")
        elif dist_pg <= 3:
            print("⚠️ Zone dangereuse - Le garde est proche du prisoner")
        elif dist_pe <= 2:
            print("🏁 Quasi-victoire! Le prisoner approche de l'évasion!")
        elif dist_pe <= 4:
            print("🎯 Zone critique - Le prisoner approche de son objectif")
        else:
            print("🔄 Phase de positionnement - Agents en recherche de position")
        
        # Prédiction
        if dist_pg > 0 and dist_pe > 0:
            if dist_pe < dist_pg:
                print("📈 Avantage: PRISONER (plus proche de l'évasion)")
            elif dist_pg < 3:
                print("📈 Avantage: GUARD (position de menace)")
            else:
                print("⚖️ Situation équilibrée")
    
    def print_game_over(self, winner, total_steps, total_rewards):
        """Affiche l'écran de fin de partie"""
        print("\n" + "=" * 60)
        print("🏁 FIN DE PARTIE")
        print("=" * 60)
        
        if winner:
            if winner == "prisoner":
                print("🎉 VICTOIRE DU PRISONER! 🟠")
                print("   Le prisoner a réussi à s'échapper!")
            else:
                print("🎉 VICTOIRE DU GUARD! 🔵")
                print("   Le garde a capturé le prisoner!")
        else:
            print("⏱️ MATCH NUL")
            print("   Limite de temps atteinte")
        
        print(f"\n📊 STATISTIQUES:")
        print(f"   Nombre de tours: {total_steps}")
        print(f"   Récompenses finales: {total_rewards}")
        
        print(f"\n🏆 RÉSUMÉ:")
        if total_steps < 50:
            print("   Partie rapide - Confrontation directe")
        elif total_steps < 100:
            print("   Partie tactique - Bon équilibre")
        else:
            print("   Partie longue - Stratégies défensives")

class ImprovedGameRunner:
    """Classe principale pour exécuter une partie avec les agents améliorés"""
    
    def __init__(self, experiment_path, models_dir="./improved_trained_models"):
        self.experiment_path = experiment_path
        self.models_dir = Path(models_dir)
        self.agents = {}
        self.visualizer = GameVisualizer()
        self.env = None
    
    def load_agents(self, episode_number=None):
        """Charge les agents d'un épisode spécifique"""
        if episode_number is None:
            # Trouve le dernier épisode disponible
            prisoner_files = list(self.models_dir.glob("prisoner_improved_episode_*.json"))
            if prisoner_files:
                episode_numbers = [int(f.stem.split('_')[-1]) for f in prisoner_files]
                episode_number = max(episode_numbers)
            else:
                raise FileNotFoundError("Aucun modèle amélioré trouvé")
        
        print(f"🔄 Chargement des agents de l'épisode {episode_number}...")
        
        # Chargement des agents
        for agent_name in ["prisoner", "guard"]:
            model_path = self.models_dir / f"{agent_name}_improved_episode_{episode_number}.json"
            if model_path.exists():
                agent = ImprovedGameAgent(agent_name)
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
    
    def play_game(self, max_steps=200, step_delay=2.0, deterministic=True):
        """Joue une partie complète avec visualisation"""
        print("🚀 Début de la partie...")
        print(f"⚙️ Mode: {'Déterministe' if deterministic else 'Stochastique'}")
        print(f"⏱️ Délai entre tours: {step_delay}s")
        print(f"🎲 Maximum {max_steps} tours")
        
        input("\n▶️ Appuyez sur Entrée pour commencer...")
        
        env = self.create_env()
        obs, infos = env.reset()
        
        total_rewards = {"prisoner": 0, "guard": 0}
        step_count = 0
        done = False
        winner = None
        
        while not done and step_count < max_steps:
            step_count += 1
            actions = {}
            strategies = {}
            confidences = {}
            decision_types = {}
            
            # Collecte des actions et analyses pour chaque agent
            for agent_name in ["prisoner", "guard"]:
                if agent_name in obs:
                    obs_array = np.array(obs[agent_name])
                    
                    # Obtention de l'action avec analyse
                    action, decision_type, confidence = self.agents[agent_name].get_action_with_analysis(
                        obs_array, deterministic
                    )
                    actions[agent_name] = int(action)
                    
                    # Analyse stratégique
                    strategy = self.agents[agent_name].get_strategy_analysis(obs_array)
                    strategies[agent_name] = strategy
                    confidences[agent_name] = confidence
                    decision_types[agent_name] = decision_type
            
            # Affichage de l'état du jeu
            self.visualizer.print_game_state(
                step_count, obs, actions, strategies, confidences, decision_types
            )
            
            # Pause pour permettre de suivre la partie
            if step_delay > 0:
                time.sleep(step_delay)
            
            # Exécution de l'étape
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Mise à jour des récompenses
            for agent_name in ["prisoner", "guard"]:
                if agent_name in rewards:
                    reward = rewards[agent_name]
                    total_rewards[agent_name] += reward
                    
                    if terminations.get(agent_name, False) and reward > 0:
                        winner = agent_name
            
            obs = next_obs
            
            # Vérification de fin de partie
            if any(terminations.values()) or any(truncations.values()):
                done = True
        
        # Affichage final
        self.visualizer.print_game_over(winner, step_count, total_rewards)
        
        return winner, step_count, total_rewards
    
    def play_multiple_games(self, num_games=3, step_delay=1.5):
        """Joue plusieurs parties consécutives"""
        print(f"🎮 Série de {num_games} parties")
        
        results = []
        
        for game_num in range(num_games):
            print(f"\n{'='*20} PARTIE {game_num + 1}/{num_games} {'='*20}")
            
            winner, steps, rewards = self.play_game(step_delay=step_delay)
            results.append({
                "game": game_num + 1,
                "winner": winner,
                "steps": steps,
                "rewards": rewards
            })
            
            if game_num < num_games - 1:
                input(f"\n⏳ Partie {game_num + 1} terminée. Appuyez sur Entrée pour la partie suivante...")
        
        # Résumé de la série
        self.print_series_summary(results)
        
        return results
    
    def print_series_summary(self, results):
        """Affiche le résumé d'une série de parties"""
        print(f"\n{'='*60}")
        print("📈 RÉSUMÉ DE LA SÉRIE")
        print(f"{'='*60}")
        
        prisoner_wins = sum(1 for r in results if r["winner"] == "prisoner")
        guard_wins = sum(1 for r in results if r["winner"] == "guard")
        draws = sum(1 for r in results if r["winner"] is None)
        
        avg_steps = np.mean([r["steps"] for r in results])
        
        print(f"🏆 Victoires:")
        print(f"   🟠 Prisoner: {prisoner_wins}/{len(results)} ({prisoner_wins/len(results)*100:.1f}%)")
        print(f"   🔵 Guard: {guard_wins}/{len(results)} ({guard_wins/len(results)*100:.1f}%)")
        print(f"   ⚖️ Match nul: {draws}/{len(results)} ({draws/len(results)*100:.1f}%)")
        
        print(f"\n📊 Statistiques:")
        print(f"   Durée moyenne: {avg_steps:.1f} tours")
        print(f"   Partie la plus courte: {min(r['steps'] for r in results)} tours")
        print(f"   Partie la plus longue: {max(r['steps'] for r in results)} tours")
        
        # Analyse de performance
        if prisoner_wins > guard_wins:
            print(f"\n🎯 Les agents prisoners sont dominants!")
        elif guard_wins > prisoner_wins:
            print(f"\n🎯 Les agents guards sont dominants!")
        else:
            print(f"\n⚖️ Les agents sont bien équilibrés!")
        
        if avg_steps < 50:
            print("⚡ Parties rapides - Stratégies agressives")
        elif avg_steps > 150:
            print("🐌 Parties longues - Stratégies défensives")
        else:
            print("🎯 Durée équilibrée - Bon compromis tactique")
    
    def close(self):
        """Ferme l'environnement"""
        if self.env:
            self.env.close()

async def main():
    """Fonction principale interactive"""
    print("🎮 PRISON ESCAPE - PARTIE AVEC AGENTS AMÉLIORÉS")
    print("=" * 55)
    
    # Configuration
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    models_dir = "./improved_trained_models"
    
    # Vérification des modèles
    models_path = Path(models_dir)
    if not models_path.exists():
        print("❌ Répertoire des modèles améliorés non trouvé!")
        print(f"   Lancez d'abord: python improved_train_prison_escape.py")
        return
    
    prisoner_files = list(models_path.glob("prisoner_improved_episode_*.json"))
    if not prisoner_files:
        print("❌ Aucun modèle amélioré trouvé!")
        print(f"   Lancez d'abord: python improved_train_prison_escape.py")
        return
    
    # Création du runner
    runner = ImprovedGameRunner(exp_path, models_dir)
    
    try:
        # Chargement des agents
        episode_number = runner.load_agents()
        print(f"✅ Agents de l'épisode {episode_number} prêts!")
        
        # Menu interactif
        while True:
            print(f"\n{'='*40}")
            print("MENU PRINCIPAL")
            print(f"{'='*40}")
            print("1. 🎮 Jouer une partie (mode normal)")
            print("2. ⚡ Jouer une partie rapide (délai court)")
            print("3. 🎲 Jouer en mode stochastique")
            print("4. 🏆 Série de 3 parties")
            print("5. 🏆 Série de 5 parties")
            print("6. ❌ Quitter")
            
            choice = input("\nVotre choix (1-6): ").strip()
            
            if choice == "1":
                runner.play_game(step_delay=2.0, deterministic=True)
            elif choice == "2":
                runner.play_game(step_delay=0.8, deterministic=True)
            elif choice == "3":
                runner.play_game(step_delay=2.0, deterministic=False)
            elif choice == "4":
                runner.play_multiple_games(num_games=3, step_delay=1.5)
            elif choice == "5":
                runner.play_multiple_games(num_games=5, step_delay=1.0)
            elif choice == "6":
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide!")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        runner.close()

if __name__ == "__main__":
    asyncio.run(main())
