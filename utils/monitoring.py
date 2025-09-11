#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import torch
from config import config


class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            'rewards': [],
            'losses': [],
            'accuracies': [],
            'steps': []
        }
        self.start_time = datetime.now()
        
        # Création du dossier de logs si nécessaire
        os.makedirs(config.output_dir, exist_ok=True)
    
    def log_metrics(self, metrics_dict, step):
        """Enregistre les métriques d'entraînement"""
        # Stockage des métriques
        if 'reward' in metrics_dict:
            self.metrics['rewards'].append(metrics_dict['reward'])
            self.metrics['steps'].append(step)
        
        # Affichage console
        print(f"📊 Step {step}: Reward={metrics_dict.get('reward', 'N/A'):.3f}")
        
        # Sauvegarde périodique
        if step % config.logging_steps == 0:
            self._save_metrics()
            self.plot_training_progress()
            self.log_memory_usage()  # Surveillance mémoire
    
    def _save_metrics(self):
        """Sauvegarde les métriques dans un fichier"""
        metrics_path = os.path.join(config.output_dir, 'training_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_training_progress(self):
        """Génère des graphiques de progression de l'entraînement"""
        if len(self.metrics['rewards']) < 2:
            return  # Pas assez de données
        
        plt.figure(figsize=(15, 5))
        
        # Graphique des récompenses
        plt.subplot(131)
        plt.plot(self.metrics['steps'], self.metrics['rewards'], 'b-', alpha=0.7)
        plt.title('Évolution des Récompenses')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Graphique des récompenses mobiles (moyenne sur 10 steps)
        if len(self.metrics['rewards']) > 10:
            plt.subplot(132)
            moving_avg = np.convolve(self.metrics['rewards'], np.ones(10)/10, mode='valid')
            plt.plot(self.metrics['steps'][9:], moving_avg, 'r-', linewidth=2)
            plt.title('Récompense Moyenne (10 steps)')
            plt.xlabel('Step')
            plt.ylabel('Moving Average Reward')
            plt.grid(True, alpha=0.3)
        
        # Graphique de distribution des récompenses
        plt.subplot(133)
        plt.hist(self.metrics['rewards'], bins=20, alpha=0.7, edgecolor='black', color='green')
        plt.title('Distribution des Récompenses')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(config.output_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Graphique sauvegardé: {plot_path}")
    
    def log_memory_usage(self):
        """Log l'utilisation mémoire GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"💾 GPU Memory:")
            print(f"   - Actuel: {allocated:.2f}GB")
            print(f"   - Réservé: {reserved:.2f}GB") 
            print(f"   - Max: {max_allocated:.2f}GB")
            
            # Vérification contrainte 16GB VRAM
            if max_allocated > 16:
                print("⚠️  ATTENTION: Dépassement de la limite de 16GB VRAM!")
            else:
                print(f"✅ Respect de la limite: {max_allocated:.2f}GB / 16GB")
        else:
            print("📊 Mode CPU - pas de monitoring GPU mémoire")
    
    def final_report(self):
        """Génère un rapport final d'entraînement"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("📋 RAPPORT FINAL D'ENTRAÎNEMENT - TEST BNP PARIBAS")
        print("="*60)
        print(f"⏱️  Durée totale: {total_time/60:.1f} minutes")
        print(f"📊 Steps complétés: {len(self.metrics['steps'])}")
        
        if self.metrics['rewards']:
            avg_reward = np.mean(self.metrics['rewards'])
            max_reward = np.max(self.metrics['rewards'])
            std_reward = np.std(self.metrics['rewards'])
            
            print(f"🏆 Récompense moyenne: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"🎯 Récompense maximale: {max_reward:.3f}")
            
            # Calcul du pourcentage de réponses correctes
            correct_answers = sum(1 for r in self.metrics['rewards'] if r > 0.5)
            accuracy = correct_answers / len(self.metrics['rewards']) if self.metrics['rewards'] else 0
            print(f"✅ Exactitude: {accuracy:.2%} ({correct_answers}/{len(self.metrics['rewards'])})")
        
        # Rapport mémoire final
        self.log_memory_usage()
        
        # Sauvegarde finale
        self._save_metrics()
        self.plot_training_progress()
        
        print(f"💾 Données sauvegardées dans: {config.output_dir}")
        print("="*60)

# Pour tester le monitoring
if __name__ == "__main__":
    print("🧪 Test du système de monitoring...")
    monitor = TrainingMonitor()
    
    # Test avec des données simulées
    for i in range(50):
        mock_metrics = {'reward': np.random.uniform(-0.5, 1.0)}
        monitor.log_metrics(mock_metrics, i)
    
    monitor.final_report()
    print("✅ Test du monitoring terminé avec succès!")