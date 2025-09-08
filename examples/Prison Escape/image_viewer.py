import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import time
import threading
from pathlib import Path
from collections import deque
from queue import Queue
import logging

class ImageViewer:
    def __init__(self, snapshot_dir):
        self.snapshot_dir = Path(snapshot_dir)
        self.root = tk.Tk()
        self.root.title("GAMA Prison Escape - Live Viewer")
        self.root.geometry("800x600")
        
        # Variables
        self.current_image = None
        self.image_label = None
        self.last_image_number = -1
        self.running = True
        self.threads_started = False
        
        # Buffer d'images (stocke les images préchargées)
        self.image_buffer = {}  # {numero: ImageTk.PhotoImage}
        self.max_buffer_size = 10  # Limite du buffer
        
        # Queue pour la communication entre threads
        self.image_queue = Queue()
        self.update_queue = Queue()
        
        # References aux threads
        self.file_monitor_thread = None
        self.image_loader_thread = None
        
        # Statistiques
        self.stats = {
            'images_loaded': 0,
            'buffer_hits': 0,
            'buffer_misses': 0
        }
        
        # Interface
        self.setup_ui()
        
        # Démarrer les threads
        self.start_threads()
        
        # Démarrer le processus de mise à jour de l'interface
        self.process_updates()
        
        # Charger la première image disponible
        self.load_initial_image()
    
    def start_threads(self):
        """Démarre les threads de surveillance et de chargement"""
        if not self.threads_started and self.running:
            self.file_monitor_thread = threading.Thread(target=self.monitor_folder, daemon=True)
            self.image_loader_thread = threading.Thread(target=self.image_loader, daemon=True)
            
            self.file_monitor_thread.start()
            self.image_loader_thread.start()
            self.threads_started = True
            print("Threads démarrés avec succès")
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Titre
        title_label = ttk.Label(main_frame, text="Prison Escape - Visualisation en temps réel", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Label pour l'image
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=1, column=0, pady=10)
        
        # Frame pour les statistiques
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=2, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Label pour le statut
        self.status_label = ttk.Label(stats_frame, text="En attente d'images...", 
                                     font=("Arial", 10))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Label pour les statistiques
        self.stats_label = ttk.Label(stats_frame, text="", 
                                    font=("Arial", 8))
        self.stats_label.grid(row=1, column=0, sticky=tk.W)
        
        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        
    def get_all_image_numbers(self):
        """Récupère tous les numéros d'images disponibles, triés"""
        if not self.snapshot_dir.exists():
            return []
            
        png_files = list(self.snapshot_dir.glob("frame*.png"))
        numbers = []
        
        for filepath in png_files:
            try:
                # Extraire le numéro du nom "frame123.png"
                filename = filepath.stem  # "frame123"
                if filename.startswith("frame"):
                    number_str = filename[5:]  # "123"
                    numbers.append(int(number_str))
            except ValueError:
                continue
                
        return sorted(numbers)
    
    def get_latest_image_number(self):
        """Trouve le numéro de la dernière image"""
        numbers = self.get_all_image_numbers()
        return numbers[-1] if numbers else -1
    
    def preload_image(self, image_number):
        """Précharge une image dans le buffer"""
        if not self.running:  # Vérifier si on doit s'arrêter
            return False
            
        if image_number in self.image_buffer:
            return True
            
        image_path = self.snapshot_dir / f"frame{image_number}.png"
        if not image_path.exists():
            return False
            
        try:
            # Vérifier la taille du fichier pour éviter les fichiers corrompus
            if image_path.stat().st_size == 0:
                print(f"Fichier vide ignoré: {image_path}")
                return False
            
            # Charger l'image avec timeout
            pil_image = Image.open(image_path)
            
            # Vérifier que l'image n'est pas corrompue
            pil_image.verify()
            
            # Recharger l'image car verify() la corrompt
            pil_image = Image.open(image_path)
            
            # Redimensionner l'image
            window_width = 700
            window_height = 500
            
            img_width, img_height = pil_image.size
            if img_width == 0 or img_height == 0:
                print(f"Image avec dimensions invalides: {image_path}")
                return False
                
            ratio = min(window_width/img_width, window_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            if new_width <= 0 or new_height <= 0:
                print(f"Dimensions calculées invalides: {new_width}x{new_height}")
                return False
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convertir pour tkinter
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Ajouter au buffer seulement si on est toujours en cours d'exécution
            if self.running:
                self.image_buffer[image_number] = tk_image
                self.stats['images_loaded'] += 1
                
                # Nettoyer le buffer si trop plein
                self.cleanup_buffer()
            
            return True
            
        except FileNotFoundError:
            print(f"Fichier non trouvé: {image_path}")
            return False
        except OSError as e:
            print(f"Erreur OS lors du chargement de l'image {image_number}: {e}")
            return False
        except Exception as e:
            print(f"Erreur lors du préchargement de l'image {image_number}: {e}")
            return False
    
    def cleanup_buffer(self):
        """Nettoie le buffer pour éviter l'accumulation en mémoire"""
        if len(self.image_buffer) <= self.max_buffer_size:
            return
            
        # Garder seulement les images les plus récentes
        sorted_numbers = sorted(self.image_buffer.keys(), reverse=True)
        to_remove = sorted_numbers[self.max_buffer_size:]
        
        for num in to_remove:
            del self.image_buffer[num]
        
    def display_image(self, image_number):
        """Affiche une image à partir du buffer"""
        if image_number in self.image_buffer:
            self.current_image = self.image_buffer[image_number]
            self.image_label.configure(image=self.current_image)
            self.last_image_number = image_number
            self.stats['buffer_hits'] += 1
            
            # Mettre à jour le statut
            status_text = f"Image: frame{image_number}.png - {time.strftime('%H:%M:%S')}"
            self.update_queue.put(('status', status_text))
            
            # Mettre à jour les statistiques
            buffer_size = len(self.image_buffer)
            stats_text = f"Buffer: {buffer_size}/{self.max_buffer_size} | Hits: {self.stats['buffer_hits']} | Misses: {self.stats['buffer_misses']} | Loaded: {self.stats['images_loaded']}"
            self.update_queue.put(('stats', stats_text))
            
            return True
        else:
            self.stats['buffer_misses'] += 1
            return False
    
    def load_initial_image(self):
        """Charge la première image disponible"""
        latest_num = self.get_latest_image_number()
        if latest_num >= 0:
            if self.preload_image(latest_num):
                self.display_image(latest_num)
    
    def image_loader(self):
        """Thread qui précharge les images"""
        print("Thread image_loader démarré")
        while self.running:
            try:
                if not self.image_queue.empty():
                    image_number = self.image_queue.get(timeout=0.1)
                    
                    # Vérifier si on doit s'arrêter
                    if not self.running:
                        break
                    
                    # Précharger cette image et quelques suivantes potentielles
                    for i in range(3):  # Précharger 3 images à l'avance
                        if not self.running:
                            break
                        self.preload_image(image_number + i)
                    
                    # Afficher l'image si elle est plus récente
                    if image_number > self.last_image_number and self.running:
                        if self.display_image(image_number):
                            pass  # Image affichée avec succès
                        else:
                            # Réessayer après un court délai
                            time.sleep(0.01)
                            if self.running and self.preload_image(image_number):
                                self.display_image(image_number)
                
                time.sleep(0.01)
                
            except Exception as e:
                if self.running:  # Ne logger que si on n'est pas en train de fermer
                    print(f"Erreur dans image_loader: {e}")
                time.sleep(0.1)
        
        print("Thread image_loader arrêté")
                
    def monitor_folder(self):
        """Surveille le dossier pour les nouvelles images"""
        print("Thread monitor_folder démarré")
        last_check_numbers = set()
        
        while self.running:
            try:
                if not self.running:
                    break
                    
                current_numbers = set(self.get_all_image_numbers())
                new_numbers = current_numbers - last_check_numbers
                
                # Traiter les nouvelles images
                for num in sorted(new_numbers):
                    if not self.running:
                        break
                    if num > self.last_image_number:
                        try:
                            self.image_queue.put(num, timeout=0.1)
                        except:
                            pass  # Queue pleine, ignore
                
                last_check_numbers = current_numbers
                time.sleep(0.05)  # Vérifier toutes les 50ms
                
            except Exception as e:
                if self.running:  # Ne logger que si on n'est pas en train de fermer
                    print(f"Erreur dans la surveillance: {e}")
                time.sleep(0.5)
        
        print("Thread monitor_folder arrêté")
    
    def process_updates(self):
        """Traite les mises à jour de l'interface utilisateur"""
        if not self.running:
            return
            
        try:
            while not self.update_queue.empty():
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == 'status':
                    self.status_label.configure(text=data)
                elif update_type == 'stats':
                    self.stats_label.configure(text=data)
                    
        except Exception as e:
            pass  # Queue vide
        
        # Programmer la prochaine vérification seulement si on est toujours actif
        if self.running:
            self.root.after(16, self.process_updates)  # ~60 FPS
                
    def on_closing(self):
        """Gérer la fermeture de la fenêtre"""
        print("Fermeture en cours...")
        
        # Arrêter les threads
        self.running = False
        
        # Vider les queues pour débloquer les threads
        try:
            while not self.image_queue.empty():
                self.image_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.update_queue.empty():
                self.update_queue.get_nowait()
        except:
            pass
        
        # Attendre que les threads se terminent (avec timeout)
        if self.file_monitor_thread and self.file_monitor_thread.is_alive():
            print("Arrêt du thread de surveillance...")
            self.file_monitor_thread.join(timeout=2.0)
            
        if self.image_loader_thread and self.image_loader_thread.is_alive():
            print("Arrêt du thread de chargement...")
            self.image_loader_thread.join(timeout=2.0)
        
        # Nettoyer le buffer d'images
        self.image_buffer.clear()
        
        print("Fermeture terminée")
        self.root.destroy()
        
    def run(self):
        """Démarrer l'interface"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    # Chemin vers le dossier snapshot
    snapshot_directory = r"e:\Stage\gama-pettingzoo\examples\Prison Escape\snapshot"
    
    # Vérifier que le dossier existe
    if not os.path.exists(snapshot_directory):
        print(f"Erreur: Le dossier {snapshot_directory} n'existe pas.")
        print("Veuillez vous assurer que le modèle GAMA est configuré pour sauvegarder dans ce dossier.")
        return
        
    # Créer et lancer le visualiseur
    viewer = ImageViewer(snapshot_directory)
    print("Visualiseur démarré. Surveillant le dossier:", snapshot_directory)
    print("Fermez la fenêtre pour arrêter le programme.")
    viewer.run()

if __name__ == "__main__":
    main()
