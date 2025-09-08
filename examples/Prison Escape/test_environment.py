"""
Script de test simple pour vérifier le fonctionnement de l'environnement Prison Escape
Ce script teste la connectivité GAMA et l'environnement de base
"""

import asyncio
import sys
from pathlib import Path

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

async def test_basic_connection():
    """Test de connexion basique à GAMA"""
    print("=== TEST DE CONNEXION GAMA ===")
    
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    print(f"Chemin expérience: {exp_path}")
    
    try:
        env = GamaParallelEnv(
            gaml_experiment_path=exp_path,
            gaml_experiment_name="main",
            gama_ip_address="localhost",
            gama_port=1001
        )
        
        print("✓ Environnement créé avec succès")
        
        # Test de reset
        obs, infos = env.reset()
        print(f"✓ Reset réussi")
        print(f"Agents détectés: {list(obs.keys())}")
        print(f"Observations: {obs}")
        
        # Test de quelques étapes
        print("\n=== TEST DE QUELQUES ÉTAPES ===")
        for step in range(3):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, info = env.step(actions)
            
            print(f"Étape {step + 1}:")
            print(f"  Actions: {actions}")
            print(f"  Récompenses: {rewards}")
            print(f"  Terminations: {terminations}")
            print(f"  Observations: {obs}")
            
            if any(terminations.values()):
                print("  Episode terminé")
                break
        
        env.close()
        print("✓ Test terminé avec succès")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test: {e}")
        return False

async def test_action_spaces():
    """Test des espaces d'action et d'observation"""
    print("\n=== TEST DES ESPACES D'ACTION ===")
    
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    
    try:
        env = GamaParallelEnv(
            gaml_experiment_path=exp_path,
            gaml_experiment_name="main",
            gama_ip_address="localhost",
            gama_port=1001
        )
        
        obs, infos = env.reset()
        
        for agent in env.agents:
            action_space = env.action_space(agent)
            observation_space = env.observation_space(agent)
            
            print(f"Agent: {agent}")
            print(f"  Espace d'action: {action_space}")
            print(f"  Espace d'observation: {observation_space}")
            print(f"  Observation actuelle: {obs[agent]}")
            
            # Test d'actions spécifiques
            for action in range(4):  # 0, 1, 2, 3
                print(f"  Action {action} valide: {action in action_space}")
        
        env.close()
        print("✓ Test des espaces terminé")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test des espaces: {e}")
        return False

async def test_complete_episode():
    """Test d'un épisode complet"""
    print("\n=== TEST D'UN ÉPISODE COMPLET ===")
    
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    
    try:
        env = GamaParallelEnv(
            gaml_experiment_path=exp_path,
            gaml_experiment_name="main",
            gama_ip_address="localhost",
            gama_port=1001
        )
        
        obs, infos = env.reset()
        step_count = 0
        max_steps = 50
        
        print(f"Début de l'épisode (max {max_steps} étapes)")
        
        while env.agents and step_count < max_steps:
            # Actions aléatoires
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            obs, rewards, terminations, truncations, info = env.step(actions)
            step_count += 1
            
            print(f"Étape {step_count}: Actions={actions}, Récompenses={rewards}")
            
            if any(terminations.values()) or any(truncations.values()):
                print(f"Épisode terminé à l'étape {step_count}")
                if any(rewards.values()):
                    winner = max(rewards.keys(), key=lambda k: rewards[k])
                    print(f"Gagnant: {winner} (récompense: {rewards[winner]})")
                break
        
        if step_count >= max_steps:
            print(f"Épisode terminé par limite de temps ({max_steps} étapes)")
        
        env.close()
        print("✓ Test d'épisode complet terminé")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test d'épisode: {e}")
        return False

def check_file_exists():
    """Vérifie que les fichiers nécessaires existent"""
    print("=== VÉRIFICATION DES FICHIERS ===")
    
    files_to_check = [
        "controler.gaml",
        "PrisonEscape.gaml"
    ]
    
    all_exist = True
    for filename in files_to_check:
        filepath = Path(__file__).parents[0] / filename
        if filepath.exists():
            print(f"✓ {filename} trouvé")
        else:
            print(f"✗ {filename} manquant: {filepath}")
            all_exist = False
    
    return all_exist

async def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("TESTS DE L'ENVIRONNEMENT PRISON ESCAPE")
    print("=" * 60)
    
    # Vérification des fichiers
    if not check_file_exists():
        print("\n✗ Des fichiers nécessaires sont manquants. Arrêt des tests.")
        sys.exit(1)
    
    # Tests
    tests = [
        ("Connexion basique", test_basic_connection),
        ("Espaces d'action", test_action_spaces),
        ("Épisode complet", test_complete_episode)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Erreur inattendue dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ RÉUSSI" if result else "✗ ÉCHOUÉ"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests réussis: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 Tous les tests sont passés! L'environnement est prêt pour l'entraînement.")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez la configuration GAMA.")
    
    print("\nConseils:")
    print("- Assurez-vous que GAMA est lancé sur le port 1001")
    print("- Vérifiez que les fichiers .gaml sont dans le bon répertoire")
    print("- Pour l'entraînement simple: python simple_train_prison_escape.py")

if __name__ == "__main__":
    asyncio.run(main())
