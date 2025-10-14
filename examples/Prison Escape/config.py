"""
Configuration file for Prison Escape training
Contains all configurable parameters for training and evaluation
"""

import os
from pathlib import Path

class PrisonEscapeConfig:
    """Centralized configuration for Prison Escape training"""
    
    # File paths
    EXPERIMENT_PATH = str(Path(__file__).parents[0] / "controler.gaml")
    EXPERIMENT_NAME = "main"
    SAVE_DIR = "./trained_models"
    LOGS_DIR = "./training_logs"
    
    # GAMA configuration
    GAMA_IP = "localhost"
    GAMA_PORT = 1001
    
    # Training parameters
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 200
    SAVE_INTERVAL = 100  # Save every N episodes
    LOG_INTERVAL = 10    # Display stats every N episodes
    
    # Agent configuration
    AGENTS = ["prisoner", "guard"]
    
    # Observation and action spaces
    GRID_SIZE = 7
    NUM_POSITIONS = GRID_SIZE * GRID_SIZE  # 49 positions possibles
    NUM_ACTIONS = 4  # gauche, droite, haut, bas
    
    # Paramètres PPO pour le prisoner
    PRISONER_CONFIG = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,          # Coefficient d'entropie pour encourager l'exploration
        "vf_coef": 0.5,            # Value function coefficient
        "max_grad_norm": 0.5,      # Clipping du gradient
        "gae_lambda": 0.95         # GAE lambda
    }
    
    # Paramètres PPO pour le guard
    GUARD_CONFIG = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95
    }
    
    # Paramètres d'évaluation
    EVAL_EPISODES = 100
    RENDER_EPISODES = 5  # Nombre d'épisodes à afficher lors de l'évaluation
    
    # Paramètres de curriculum learning (optionnel)
    CURRICULUM_ENABLED = False
    CURRICULUM_STAGES = [
        {"episodes": 200, "max_steps": 50},   # Étape 1: épisodes courts
        {"episodes": 300, "max_steps": 100},  # Étape 2: épisodes moyens
        {"episodes": 500, "max_steps": 200}   # Étape 3: épisodes complets
    ]
    
    # Paramètres de visualisation
    PLOT_EVERY_N_EPISODES = 50
    SAVE_PLOTS = True
    PLOT_DPI = 300
    
    @classmethod
    def create_directories(cls):
        """Crée les répertoires nécessaires"""
        Path(cls.SAVE_DIR).mkdir(exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(exist_ok=True)
        print(f"Répertoires créés: {cls.SAVE_DIR}, {cls.LOGS_DIR}")
    
    @classmethod
    def get_model_config(cls, agent_name):
        """Retourne la configuration pour un agent spécifique"""
        if agent_name == "prisoner":
            return cls.PRISONER_CONFIG
        elif agent_name == "guard":
            return cls.GUARD_CONFIG
        else:
            raise ValueError(f"Agent inconnu: {agent_name}")
    
    @classmethod
    def print_config(cls):
        """Affiche la configuration actuelle"""
        print("=== CONFIGURATION PRISON ESCAPE ===")
        print(f"Expérience GAMA: {cls.EXPERIMENT_PATH}")
        print(f"Nom expérience: {cls.EXPERIMENT_NAME}")
        print(f"IP GAMA: {cls.GAMA_IP}:{cls.GAMA_PORT}")
        print(f"Nombre d'épisodes: {cls.NUM_EPISODES}")
        print(f"Étapes max par épisode: {cls.MAX_STEPS_PER_EPISODE}")
        print(f"Taille de grille: {cls.GRID_SIZE}x{cls.GRID_SIZE}")
        print(f"Nombre d'actions: {cls.NUM_ACTIONS}")
        print(f"Répertoire sauvegarde: {cls.SAVE_DIR}")
        print(f"Curriculum learning: {'Activé' if cls.CURRICULUM_ENABLED else 'Désactivé'}")
        print("=" * 40)

# Configuration par défaut pour les tests rapides
class QuickTestConfig(PrisonEscapeConfig):
    """Configuration pour des tests rapides"""
    NUM_EPISODES = 50
    MAX_STEPS_PER_EPISODE = 50
    SAVE_INTERVAL = 25
    LOG_INTERVAL = 5
    EVAL_EPISODES = 20
    
    PRISONER_CONFIG = {
        "learning_rate": 1e-3,
        "n_steps": 512,
        "batch_size": 32,
        "n_epochs": 5,
        "gamma": 0.99,
        "clip_range": 0.2
    }
    
    GUARD_CONFIG = {
        "learning_rate": 1e-3,
        "n_steps": 512,
        "batch_size": 32,
        "n_epochs": 5,
        "gamma": 0.99,
        "clip_range": 0.2
    }

# Configuration pour l'entraînement intensif
class IntensiveTrainingConfig(PrisonEscapeConfig):
    """Configuration pour un entraînement intensif"""
    NUM_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 300
    SAVE_INTERVAL = 250
    LOG_INTERVAL = 25
    EVAL_EPISODES = 200
    
    CURRICULUM_ENABLED = True
    
    PRISONER_CONFIG = {
        "learning_rate": 2e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.995,
        "clip_range": 0.15,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "gae_lambda": 0.98
    }
    
    GUARD_CONFIG = {
        "learning_rate": 2e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.995,
        "clip_range": 0.15,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "gae_lambda": 0.98
    }

def get_config(config_name="default"):
    """Retourne la configuration demandée"""
    configs = {
        "default": PrisonEscapeConfig,
        "quick": QuickTestConfig,
        "intensive": IntensiveTrainingConfig
    }
    
    if config_name not in configs:
        raise ValueError(f"Configuration inconnue: {config_name}. Disponibles: {list(configs.keys())}")
    
    return configs[config_name]
