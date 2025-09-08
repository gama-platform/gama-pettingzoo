"""
Script utilitaire pour gérer l'entraînement et l'évaluation des agents Prison Escape
Usage: python run_training.py [commande] [options]
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Import des modules locaux
from config import get_config
from train_prison_escape import PrisonEscapeTrainer

def setup_argument_parser():
    """Configure le parser d'arguments"""
    parser = argparse.ArgumentParser(description="Entraînement et évaluation des agents Prison Escape")
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande d'entraînement
    train_parser = subparsers.add_parser('train', help='Entraîner les agents')
    train_parser.add_argument('--config', '-c', default='default', 
                             choices=['default', 'quick', 'intensive'],
                             help='Configuration à utiliser')
    train_parser.add_argument('--episodes', '-e', type=int, 
                             help='Nombre d\'épisodes (remplace la config)')
    train_parser.add_argument('--save-dir', '-s', type=str,
                             help='Répertoire de sauvegarde (remplace la config)')
    train_parser.add_argument('--port', '-p', type=int, default=1001,
                             help='Port GAMA à utiliser')
    
    # Commande d'évaluation
    eval_parser = subparsers.add_parser('eval', help='Évaluer les agents entraînés')
    eval_parser.add_argument('--models-dir', '-m', type=str, default='./trained_models',
                            help='Répertoire des modèles entraînés')
    eval_parser.add_argument('--episodes', '-e', type=int, default=100,
                            help='Nombre d\'épisodes d\'évaluation')
    eval_parser.add_argument('--episode-number', '-n', type=int,
                            help='Numéro d\'épisode spécifique à charger')
    eval_parser.add_argument('--port', '-p', type=int, default=1001,
                            help='Port GAMA à utiliser')
    eval_parser.add_argument('--render', '-r', type=int, default=5,
                            help='Nombre d\'épisodes à afficher')
    
    # Commande d'information
    info_parser = subparsers.add_parser('info', help='Afficher les informations des modèles')
    info_parser.add_argument('--models-dir', '-m', type=str, default='./trained_models',
                            help='Répertoire des modèles')
    
    # Commande d'installation
    install_parser = subparsers.add_parser('install', help='Installer les dépendances')
    
    return parser

async def run_training(args):
    """Exécute l'entraînement"""
    print("=== LANCEMENT DE L'ENTRAÎNEMENT ===")
    
    # Chargement de la configuration
    config_class = get_config(args.config)
    config_class.create_directories()
    config_class.print_config()
    
    # Override des paramètres si spécifiés
    if args.episodes:
        config_class.NUM_EPISODES = args.episodes
        print(f"Nombre d'épisodes remplacé: {args.episodes}")
    
    if args.save_dir:
        config_class.SAVE_DIR = args.save_dir
        print(f"Répertoire de sauvegarde remplacé: {args.save_dir}")
    
    if args.port != 1001:
        config_class.GAMA_PORT = args.port
        print(f"Port GAMA remplacé: {args.port}")
    
    # Création du trainer
    trainer = PrisonEscapeTrainer(
        experiment_path=config_class.EXPERIMENT_PATH,
        save_dir=config_class.SAVE_DIR,
        config_class=config_class
    )
    
    # Lancement de l'entraînement
    await trainer.train(
        num_episodes=config_class.NUM_EPISODES,
        save_interval=config_class.SAVE_INTERVAL,
        log_interval=config_class.LOG_INTERVAL
    )
    
    print("Entraînement terminé avec succès!")

async def run_evaluation(args):
    """Exécute l'évaluation"""
    print("=== LANCEMENT DE L'ÉVALUATION ===")
    
    try:
        # Import dynamique pour éviter les erreurs si les dépendances ne sont pas installées
        from evaluate_agents import PrisonEscapeEvaluator
        
        # Configuration de base
        exp_path = str(Path(__file__).parents[0] / "controler.gaml")
        
        # Création de l'évaluateur
        evaluator = PrisonEscapeEvaluator(exp_path, args.models_dir)
        
        # Chargement des modèles
        episode_number = evaluator.load_models(args.episode_number)
        print(f"Modèles de l'épisode {episode_number} chargés")
        
        # Évaluation
        results = await evaluator.evaluate(
            num_episodes=args.episodes,
            render_episodes=args.render
        )
        
        # Visualisation des résultats
        evaluator.plot_evaluation_results(results)
        
        await evaluator.close()
        
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("Assurez-vous que toutes les dépendances sont installées:")
        print("pip install -r requirements_training.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
        sys.exit(1)

def show_model_info(args):
    """Affiche les informations des modèles disponibles"""
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"Répertoire des modèles non trouvé: {models_dir}")
        return
    
    print(f"=== MODÈLES DISPONIBLES DANS {models_dir} ===")
    
    # Recherche des modèles
    prisoner_models = list(models_dir.glob("prisoner_ppo_episode_*.zip"))
    guard_models = list(models_dir.glob("guard_ppo_episode_*.zip"))
    
    if not prisoner_models and not guard_models:
        print("Aucun modèle trouvé.")
        return
    
    # Extraction des numéros d'épisodes
    prisoner_episodes = sorted([int(f.stem.split('_')[-1]) for f in prisoner_models])
    guard_episodes = sorted([int(f.stem.split('_')[-1]) for f in guard_models])
    
    print(f"Modèles Prisoner: {len(prisoner_episodes)} disponibles")
    if prisoner_episodes:
        print(f"  Épisodes: {prisoner_episodes[0]} à {prisoner_episodes[-1]}")
    
    print(f"Modèles Guard: {len(guard_episodes)} disponibles")
    if guard_episodes:
        print(f"  Épisodes: {guard_episodes[0]} à {guard_episodes[-1]}")
    
    # Recherche des métriques
    metrics_files = list(models_dir.glob("training_metrics_episode_*.pkl"))
    if metrics_files:
        metrics_episodes = sorted([int(f.stem.split('_')[-1]) for f in metrics_files])
        print(f"Métriques d'entraînement: {len(metrics_episodes)} fichiers")
        print(f"  Épisodes: {metrics_episodes[0]} à {metrics_episodes[-1]}")
    
    # Recherche des graphiques
    plot_files = list(models_dir.glob("training_progress_episode_*.png"))
    if plot_files:
        plot_episodes = sorted([int(f.stem.split('_')[-1]) for f in plot_files])
        print(f"Graphiques de progression: {len(plot_episodes)} fichiers")

def install_dependencies():
    """Installe les dépendances"""
    import subprocess
    import os
    
    requirements_file = Path(__file__).parent / "requirements_training.txt"
    
    if not requirements_file.exists():
        print(f"Fichier requirements non trouvé: {requirements_file}")
        return
    
    print("Installation des dépendances...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("Dépendances installées avec succès!")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'installation: {e}")
        sys.exit(1)

async def main():
    """Fonction principale"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'train':
        await run_training(args)
    elif args.command == 'eval':
        await run_evaluation(args)
    elif args.command == 'info':
        show_model_info(args)
    elif args.command == 'install':
        install_dependencies()
    else:
        print(f"Commande inconnue: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
