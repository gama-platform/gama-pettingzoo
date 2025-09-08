"""
Évaluateur spécialisé pour les agents entraînés avec la version améliorée
Analyse les performances et comportements des modèles improved_train_prison_escape.py
"""

import asyncio
import numpy as np
import json
from pathlib import Path
import time
from collections import deque, Counter
import matplotlib.pyplot as plt

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

class ImprovedAgentEvaluator:
    """Classe pour charger et évaluer les agents améliorés"""
    
    def __init__(self, agent_name, grid_size=7):
        self.agent_name = agent_name
        self.grid_size = grid_size
        self.action_count = 4
        self.q_table = {}
        
        # Métriques de comportement
        self.action_diversity = 0
        self.exploration_level = 0
        self.states_learned = 0
        
    def get_state_key(self, observation):
        """Convertit l'observation en clé d'état"""
        return tuple(observation)
    
    def get_action(self, observation, deterministic=True, epsilon=0.05):
        """Choisit une action basée sur l'observation"""
        state_key = self.get_state_key(observation)
        
        if state_key not in self.q_table:
            # État non vu, action aléatoire
            return np.random.randint(0, self.action_count)
        
        if deterministic:
            # Mode déterministe : meilleure action
            return np.argmax(self.q_table[state_key])
        else:
            # Mode stochastique avec epsilon d'exploration
            if np.random.random() < epsilon:
                return np.random.randint(0, self.action_count)
            else:
                return np.argmax(self.q_table[state_key])
    
    def analyze_q_table(self):
        """Analyse la qualité et la diversité de la table Q"""
        if not self.q_table:
            return
        
        self.states_learned = len(self.q_table)
        
        # Analyse de la diversité des actions préférées
        preferred_actions = []
        q_values_stats = []
        
        for state, q_vals in self.q_table.items():
            preferred_actions.append(np.argmax(q_vals))
            q_values_stats.extend(q_vals)
        
        # Calcul de la diversité d'actions
        action_counts = Counter(preferred_actions)
        total_states = len(preferred_actions)
        
        if total_states > 0:
            # Entropie des actions préférées (mesure de diversité)
            probabilities = [count/total_states for count in action_counts.values()]
            self.action_diversity = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Niveau d'exploration (variance des valeurs Q)
        if q_values_stats:
            self.exploration_level = np.std(q_values_stats)
        
        print(f"  Analyse {self.agent_name}:")
        print(f"    États appris: {self.states_learned}")
        print(f"    Diversité d'actions: {self.action_diversity:.2f}/2.00")
        print(f"    Niveau d'exploration: {self.exploration_level:.3f}")
        print(f"    Distribution des actions préférées: {dict(action_counts)}")
    
    def load(self, filepath):
        """Charge l'agent depuis un fichier JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.agent_name = data['agent_name']
        self.q_table = {eval(k): np.array(v) for k, v in data['q_table'].items()}
        
        # Analyse de la table Q chargée
        self.analyze_q_table()
        
        print(f"Agent {self.agent_name} amélioré chargé depuis {filepath}")

class ImprovedEvaluator:
    """Évaluateur pour les agents entraînés avec la version améliorée"""
    
    def __init__(self, experiment_path, models_dir="./improved_trained_models"):
        self.experiment_path = experiment_path
        self.models_dir = Path(models_dir)
        self.agents = {}
        self.env = None
        
        # Métriques d'évaluation avancées
        self.behavior_metrics = {
            "action_sequences": {"prisoner": [], "guard": []},
            "position_patterns": {"prisoner": [], "guard": []},
            "interaction_frequency": 0,
            "strategic_moves": {"prisoner": 0, "guard": 0},
            "defensive_moves": {"prisoner": 0, "guard": 0}
        }
    
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
        
        print(f"\n=== CHARGEMENT DES MODÈLES AMÉLIORÉS (Épisode {episode_number}) ===")
        
        # Chargement des agents
        for agent_name in ["prisoner", "guard"]:
            model_path = self.models_dir / f"{agent_name}_improved_episode_{episode_number}.json"
            if model_path.exists():
                agent = ImprovedAgentEvaluator(agent_name)
                agent.load(model_path)
                self.agents[agent_name] = agent
            else:
                raise FileNotFoundError(f"Modèle amélioré non trouvé: {model_path}")
        
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
    
    def analyze_strategic_behavior(self, obs, actions, step_count):
        """Analyse le comportement stratégique des agents"""
        if "prisoner" not in obs:
            return
        
        p_pos = obs["prisoner"][0]
        g_pos = obs["prisoner"][1]
        e_pos = obs["prisoner"][2]
        
        # Conversion en coordonnées
        p_x, p_y = p_pos % 7, p_pos // 7
        g_x, g_y = g_pos % 7, g_pos // 7
        e_x, e_y = e_pos % 7, e_pos // 7
        
        # Calcul des distances
        p_to_escape = abs(p_x - e_x) + abs(p_y - e_y)
        g_to_prisoner = abs(g_x - p_x) + abs(g_y - p_y)
        g_to_escape = abs(g_x - e_x) + abs(g_y - e_y)
        
        # Détection d'interactions
        if g_to_prisoner <= 2:
            self.behavior_metrics["interaction_frequency"] += 1
        
        # Analyse des mouvements stratégiques du prisoner
        p_action = actions.get("prisoner", 0)
        if p_to_escape > 0:  # Si pas encore à l'évasion
            # Mouvement vers l'évasion
            if (p_action == 0 and p_x > e_x) or \
               (p_action == 1 and p_x < e_x) or \
               (p_action == 2 and p_y > e_y) or \
               (p_action == 3 and p_y < e_y):
                self.behavior_metrics["strategic_moves"]["prisoner"] += 1
            # Mouvement défensif (s'éloigner du guard)
            elif g_to_prisoner <= 3:
                if (p_action == 0 and g_x > p_x) or \
                   (p_action == 1 and g_x < p_x) or \
                   (p_action == 2 and g_y > p_y) or \
                   (p_action == 3 and g_y < p_y):
                    self.behavior_metrics["defensive_moves"]["prisoner"] += 1
        
        # Analyse des mouvements stratégiques du guard
        g_action = actions.get("guard", 0)
        if g_to_prisoner > 0:  # Si pas encore capturé le prisoner
            # Mouvement vers le prisoner
            if (g_action == 0 and g_x > p_x) or \
               (g_action == 1 and g_x < p_x) or \
               (g_action == 2 and g_y > p_y) or \
               (g_action == 3 and g_y < p_y):
                self.behavior_metrics["strategic_moves"]["guard"] += 1
            # Mouvement de blocage (vers l'évasion)
            elif g_to_escape > g_to_prisoner:
                if (g_action == 0 and g_x > e_x) or \
                   (g_action == 1 and g_x < e_x) or \
                   (g_action == 2 and g_y > e_y) or \
                   (g_action == 3 and g_y < e_y):
                    self.behavior_metrics["defensive_moves"]["guard"] += 1
        
        # Stockage des séquences d'actions
        for agent_name in ["prisoner", "guard"]:
            if agent_name in actions:
                self.behavior_metrics["action_sequences"][agent_name].append(actions[agent_name])
                # Garder seulement les 20 dernières actions
                if len(self.behavior_metrics["action_sequences"][agent_name]) > 20:
                    self.behavior_metrics["action_sequences"][agent_name].pop(0)
        
        # Stockage des patterns de position
        self.behavior_metrics["position_patterns"]["prisoner"].append(p_pos)
        self.behavior_metrics["position_patterns"]["guard"].append(g_pos)
        if len(self.behavior_metrics["position_patterns"]["prisoner"]) > 50:
            self.behavior_metrics["position_patterns"]["prisoner"].pop(0)
            self.behavior_metrics["position_patterns"]["guard"].pop(0)
    
    def reset_behavior_metrics(self):
        """Remet à zéro les métriques comportementales"""
        self.behavior_metrics = {
            "action_sequences": {"prisoner": [], "guard": []},
            "position_patterns": {"prisoner": [], "guard": []},
            "interaction_frequency": 0,
            "strategic_moves": {"prisoner": 0, "guard": 0},
            "defensive_moves": {"prisoner": 0, "guard": 0}
        }
    
    def run_evaluation_episode(self, max_steps=200, verbose=False, epsilon=0.0):
        """Exécute un épisode d'évaluation avec analyse comportementale"""
        env = self.create_env()
        obs, infos = env.reset()
        
        # Reset des métriques pour cet épisode
        self.reset_behavior_metrics()
        
        total_rewards = {"prisoner": 0, "guard": 0}
        step_count = 0
        done = False
        winner = None
        
        episode_log = {
            "steps": [],
            "final_metrics": {},
            "behavior_summary": {}
        }
        
        while not done and step_count < max_steps:
            actions = {}
            step_info = {"step": step_count + 1}
            
            # Collecte des actions
            for agent_name in ["prisoner", "guard"]:
                if agent_name in obs:
                    obs_array = np.array(obs[agent_name])
                    # Utilisation de l'epsilon pour l'évaluation (0.0 = déterministe)
                    action = self.agents[agent_name].get_action(obs_array, 
                                                               deterministic=(epsilon == 0.0),
                                                               epsilon=epsilon)
                    actions[agent_name] = int(action)
                    
                    step_info[f"{agent_name}_obs"] = obs_array.tolist()
                    step_info[f"{agent_name}_action"] = action
            
            # Analyse du comportement stratégique
            self.analyze_strategic_behavior(obs, actions, step_count)
            
            if verbose:
                self.print_step_info_detailed(step_count + 1, obs, actions)
            
            # Exécution de l'étape
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            step_info["rewards"] = rewards
            step_info["terminations"] = terminations
            episode_log["steps"].append(step_info)
            
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
        
        # Calcul des métriques finales de l'épisode
        episode_log["final_metrics"] = self.calculate_episode_metrics(step_count)
        episode_log["behavior_summary"] = self.summarize_behavior()
        
        return total_rewards, step_count, winner, episode_log
    
    def calculate_episode_metrics(self, total_steps):
        """Calcule les métriques de performance de l'épisode"""
        metrics = {}
        
        # Fréquence d'interaction
        metrics["interaction_rate"] = self.behavior_metrics["interaction_frequency"] / total_steps if total_steps > 0 else 0
        
        # Ratio de mouvements stratégiques vs défensifs
        for agent in ["prisoner", "guard"]:
            strategic = self.behavior_metrics["strategic_moves"][agent]
            defensive = self.behavior_metrics["defensive_moves"][agent]
            total_moves = strategic + defensive
            
            if total_moves > 0:
                metrics[f"{agent}_strategic_ratio"] = strategic / total_moves
                metrics[f"{agent}_defensive_ratio"] = defensive / total_moves
            else:
                metrics[f"{agent}_strategic_ratio"] = 0
                metrics[f"{agent}_defensive_ratio"] = 0
        
        # Diversité d'actions dans l'épisode
        for agent in ["prisoner", "guard"]:
            actions = self.behavior_metrics["action_sequences"][agent]
            if actions:
                unique_actions = len(set(actions))
                metrics[f"{agent}_action_diversity"] = unique_actions / 4.0  # Normalisé sur 4 actions possibles
            else:
                metrics[f"{agent}_action_diversity"] = 0
        
        # Couverture spatiale
        for agent in ["prisoner", "guard"]:
            positions = self.behavior_metrics["position_patterns"][agent]
            if positions:
                unique_positions = len(set(positions))
                metrics[f"{agent}_spatial_coverage"] = unique_positions / 49.0  # Normalisé sur 49 positions possibles
            else:
                metrics[f"{agent}_spatial_coverage"] = 0
        
        return metrics
    
    def summarize_behavior(self):
        """Résume le comportement observé dans l'épisode"""
        summary = {}
        
        # Patterns d'actions les plus fréquents
        for agent in ["prisoner", "guard"]:
            actions = self.behavior_metrics["action_sequences"][agent]
            if actions:
                action_counter = Counter(actions)
                most_common = action_counter.most_common(2)
                summary[f"{agent}_preferred_actions"] = most_common
            
            # Détection de patterns répétitifs
            if len(actions) >= 6:
                # Recherche de séquences répétitives de 3 actions
                sequences = [tuple(actions[i:i+3]) for i in range(len(actions)-2)]
                sequence_counter = Counter(sequences)
                if sequence_counter:
                    most_common_seq = sequence_counter.most_common(1)[0]
                    if most_common_seq[1] > 1:  # Répété au moins 2 fois
                        summary[f"{agent}_repetitive_pattern"] = {
                            "sequence": most_common_seq[0],
                            "frequency": most_common_seq[1]
                        }
        
        return summary
    
    def print_step_info_detailed(self, step, obs, actions):
        """Affiche les informations détaillées d'une étape"""
        if "prisoner" in obs:
            p_pos = obs["prisoner"][0]
            g_pos = obs["prisoner"][1]
            e_pos = obs["prisoner"][2]
            
            # Conversion en coordonnées (x, y)
            p_x, p_y = p_pos % 7, p_pos // 7
            g_x, g_y = g_pos % 7, g_pos // 7
            e_x, e_y = e_pos % 7, e_pos // 7
            
            action_names = ["Gauche", "Droite", "Haut", "Bas"]
            
            # Calcul des distances pour l'analyse
            p_to_escape = abs(p_x - e_x) + abs(p_y - e_y)
            g_to_prisoner = abs(g_x - p_x) + abs(g_y - p_y)
            
            print(f"Étape {step}:")
            print(f"  Prisoner: ({p_x}, {p_y}) -> {action_names[actions.get('prisoner', 0)]} [Dist. évasion: {p_to_escape}]")
            print(f"  Guard: ({g_x}, {g_y}) -> {action_names[actions.get('guard', 0)]} [Dist. prisoner: {g_to_prisoner}]")
            print(f"  Escape: ({e_x}, {e_y})")
    
    def evaluate_comprehensive(self, num_episodes=50, verbose_episodes=2, test_scenarios=True):
        """Évaluation complète avec différents scénarios"""
        print("=== ÉVALUATION COMPLÈTE DES AGENTS AMÉLIORÉS ===")
        
        results = {
            "deterministic": {"rewards": {"prisoner": [], "guard": []}, "lengths": [], "winners": {"prisoner": 0, "guard": 0, "draw": 0}, "metrics": []},
            "stochastic": {"rewards": {"prisoner": [], "guard": []}, "lengths": [], "winners": {"prisoner": 0, "guard": 0, "draw": 0}, "metrics": []}
        }
        
        scenarios = [
            ("deterministic", 0.0, "Déterministe (exploitation pure)"),
            ("stochastic", 0.1, "Stochastique (10% exploration)")
        ]
        
        for scenario_name, epsilon, description in scenarios:
            print(f"\n--- {description} ---")
            
            for episode in range(num_episodes):
                verbose = episode < verbose_episodes and scenario_name == "deterministic"
                
                if verbose:
                    print(f"\n=== ÉPISODE {episode + 1} ({description}) ===")
                
                total_rewards, episode_length, winner, episode_log = self.run_evaluation_episode(
                    verbose=verbose, 
                    epsilon=epsilon
                )
                
                # Stockage des résultats
                results[scenario_name]["rewards"]["prisoner"].append(total_rewards["prisoner"])
                results[scenario_name]["rewards"]["guard"].append(total_rewards["guard"])
                results[scenario_name]["lengths"].append(episode_length)
                results[scenario_name]["metrics"].append(episode_log["final_metrics"])
                
                if winner:
                    results[scenario_name]["winners"][winner] += 1
                else:
                    results[scenario_name]["winners"]["draw"] += 1
                
                if verbose:
                    print(f"Résultat: Gagnant={winner or 'Match nul'}, Étapes={episode_length}")
                    print(f"Récompenses={total_rewards}")
                    self.print_episode_analysis(episode_log)
                
                if (episode + 1) % 10 == 0:
                    print(f"  {scenario_name.capitalize()}: {episode + 1}/{num_episodes} épisodes terminés")
        
        self.print_comprehensive_results(results, num_episodes)
        return results
    
    def print_episode_analysis(self, episode_log):
        """Affiche l'analyse détaillée d'un épisode"""
        metrics = episode_log["final_metrics"]
        behavior = episode_log["behavior_summary"]
        
        print(f"Métriques de l'épisode:")
        print(f"  Taux d'interaction: {metrics.get('interaction_rate', 0):.2%}")
        print(f"  Prisoner - Stratégique: {metrics.get('prisoner_strategic_ratio', 0):.2%}, Défensif: {metrics.get('prisoner_defensive_ratio', 0):.2%}")
        print(f"  Guard - Stratégique: {metrics.get('guard_strategic_ratio', 0):.2%}, Défensif: {metrics.get('guard_defensive_ratio', 0):.2%}")
        print(f"  Diversité d'actions - P: {metrics.get('prisoner_action_diversity', 0):.2%}, G: {metrics.get('guard_action_diversity', 0):.2%}")
        print(f"  Couverture spatiale - P: {metrics.get('prisoner_spatial_coverage', 0):.2%}, G: {metrics.get('guard_spatial_coverage', 0):.2%}")
        
        if behavior:
            print(f"Patterns comportementaux:")
            for key, value in behavior.items():
                if "preferred_actions" in key:
                    agent = key.split("_")[0]
                    action_names = ["Gauche", "Droite", "Haut", "Bas"]
                    prefs = [f"{action_names[action]}({count})" for action, count in value]
                    print(f"  {agent.capitalize()} préfère: {', '.join(prefs)}")
                elif "repetitive_pattern" in key:
                    agent = key.split("_")[0]
                    pattern = value["sequence"]
                    freq = value["frequency"]
                    action_names = ["Gauche", "Droite", "Haut", "Bas"]
                    pattern_str = "->".join([action_names[a] for a in pattern])
                    print(f"  {agent.capitalize()} répète: {pattern_str} ({freq}x)")
    
    def print_comprehensive_results(self, results, num_episodes):
        """Affiche les résultats complets de l'évaluation"""
        print(f"\n=== RÉSULTATS COMPLETS ({num_episodes} épisodes par scénario) ===")
        
        for scenario in ["deterministic", "stochastic"]:
            data = results[scenario]
            print(f"\n--- Scénario {scenario.upper()} ---")
            
            # Statistiques de base
            p_rewards = data["rewards"]["prisoner"]
            g_rewards = data["rewards"]["guard"]
            lengths = data["lengths"]
            
            print(f"Récompense moyenne prisoner: {np.mean(p_rewards):.3f} ± {np.std(p_rewards):.3f}")
            print(f"Récompense moyenne guard: {np.mean(g_rewards):.3f} ± {np.std(g_rewards):.3f}")
            print(f"Longueur moyenne: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            
            # Répartition des victoires
            print(f"Victoires:")
            for agent, count in data["winners"].items():
                percentage = count / num_episodes * 100
                print(f"  {agent.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Moyennes des métriques comportementales
            if data["metrics"]:
                metrics_avg = {}
                for metric_key in data["metrics"][0].keys():
                    values = [m[metric_key] for m in data["metrics"] if metric_key in m]
                    if values:
                        metrics_avg[metric_key] = np.mean(values)
                
                print(f"Métriques comportementales moyennes:")
                print(f"  Taux d'interaction: {metrics_avg.get('interaction_rate', 0):.2%}")
                print(f"  Stratégie prisoner: {metrics_avg.get('prisoner_strategic_ratio', 0):.2%}")
                print(f"  Stratégie guard: {metrics_avg.get('guard_strategic_ratio', 0):.2%}")
                print(f"  Diversité prisoner: {metrics_avg.get('prisoner_action_diversity', 0):.2%}")
                print(f"  Diversité guard: {metrics_avg.get('guard_action_diversity', 0):.2%}")
                print(f"  Couverture prisoner: {metrics_avg.get('prisoner_spatial_coverage', 0):.2%}")
                print(f"  Couverture guard: {metrics_avg.get('guard_spatial_coverage', 0):.2%}")
        
        # Comparaison entre scénarios
        self.compare_scenarios(results, num_episodes)
    
    def compare_scenarios(self, results, num_episodes):
        """Compare les différents scénarios d'évaluation"""
        print(f"\n=== COMPARAISON DES SCÉNARIOS ===")
        
        det = results["deterministic"]
        sto = results["stochastic"]
        
        # Comparaison des longueurs d'épisode
        det_length = np.mean(det["lengths"])
        sto_length = np.mean(sto["lengths"])
        print(f"Longueur moyenne - Déterministe: {det_length:.1f}, Stochastique: {sto_length:.1f}")
        
        # Comparaison des taux de victoire
        det_draws = det["winners"]["draw"] / num_episodes
        sto_draws = sto["winners"]["draw"] / num_episodes
        print(f"Taux de match nul - Déterministe: {det_draws:.1%}, Stochastique: {sto_draws:.1%}")
        
        # Recommandations
        print(f"\n=== RECOMMANDATIONS ===")
        if det_draws > 0.8:
            print("⚠️  Taux de match nul élevé en mode déterministe")
            print("   -> Les agents ont convergé vers des stratégies défensives")
        
        if sto_draws < det_draws:
            print("✅ L'exploration stochastique améliore la diversité des résultats")
        
        if sto_length < det_length:
            print("✅ L'exploration permet des épisodes plus courts (plus d'interactions)")
        
        # Calcul d'un score de qualité global
        det_quality = self.calculate_behavior_quality(det)
        sto_quality = self.calculate_behavior_quality(sto)
        
        print(f"\nScore de qualité comportementale:")
        print(f"  Déterministe: {det_quality:.2f}/1.00")
        print(f"  Stochastique: {sto_quality:.2f}/1.00")
        
        if sto_quality > det_quality:
            print("🎯 L'entraînement amélioré bénéficie de l'exploration stochastique")
        else:
            print("🎯 L'entraînement amélioré a produit des stratégies stables")
    
    def calculate_behavior_quality(self, scenario_data):
        """Calcule un score de qualité comportementale"""
        if not scenario_data["metrics"]:
            return 0.0
        
        metrics_avg = {}
        for metric_key in scenario_data["metrics"][0].keys():
            values = [m[metric_key] for m in scenario_data["metrics"] if metric_key in m]
            if values:
                metrics_avg[metric_key] = np.mean(values)
        
        # Score composite (0 à 1)
        score = 0.0
        
        # Diversité d'actions (0.3 du score)
        action_div = (metrics_avg.get('prisoner_action_diversity', 0) + 
                     metrics_avg.get('guard_action_diversity', 0)) / 2
        score += action_div * 0.3
        
        # Couverture spatiale (0.2 du score)
        spatial_cov = (metrics_avg.get('prisoner_spatial_coverage', 0) + 
                      metrics_avg.get('guard_spatial_coverage', 0)) / 2
        score += spatial_cov * 0.2
        
        # Équilibre stratégique/défensif (0.3 du score)
        p_balance = min(metrics_avg.get('prisoner_strategic_ratio', 0),
                       metrics_avg.get('prisoner_defensive_ratio', 0)) * 2
        g_balance = min(metrics_avg.get('guard_strategic_ratio', 0),
                       metrics_avg.get('guard_defensive_ratio', 0)) * 2
        score += (p_balance + g_balance) / 2 * 0.3
        
        # Taux d'interaction (0.2 du score)
        interaction = min(1.0, metrics_avg.get('interaction_rate', 0) * 5)  # Cap à 1.0
        score += interaction * 0.2
        
        return min(1.0, score)
    
    def close(self):
        """Ferme l'environnement"""
        if self.env:
            self.env.close()

def list_improved_models(models_dir="./improved_trained_models"):
    """Liste les modèles améliorés disponibles"""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Répertoire non trouvé: {models_dir}")
        return []
    
    prisoner_files = list(models_path.glob("prisoner_improved_episode_*.json"))
    if not prisoner_files:
        print("Aucun modèle amélioré trouvé")
        return []
    
    episodes = sorted([int(f.stem.split('_')[-1]) for f in prisoner_files])
    print(f"Modèles améliorés disponibles: épisodes {episodes}")
    return episodes

async def main():
    """Fonction principale"""
    print("=== ÉVALUATEUR D'AGENTS PRISON ESCAPE AMÉLIORÉS ===")
    
    # Configuration
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    models_dir = "./improved_trained_models"
    
    # Liste des modèles disponibles
    available_episodes = list_improved_models(models_dir)
    if not available_episodes:
        print("⚠️  Aucun modèle amélioré trouvé.")
        print("   Lancez d'abord: python improved_train_prison_escape.py")
        return
    
    # Création de l'évaluateur
    evaluator = ImprovedEvaluator(exp_path, models_dir)
    
    try:
        # Chargement du dernier modèle
        episode_number = evaluator.load_agents()
        print(f"\n✅ Modèles améliorés de l'épisode {episode_number} chargés")
        
        # Évaluation complète
        print(f"\n🚀 Début de l'évaluation complète...")
        results = evaluator.evaluate_comprehensive(
            num_episodes=25,  # 25 épisodes par scénario
            verbose_episodes=2,
            test_scenarios=True
        )
        
        print(f"\n🎉 Évaluation terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        evaluator.close()

if __name__ == "__main__":
    asyncio.run(main())
