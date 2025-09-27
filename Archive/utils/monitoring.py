#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training monitoring and visualization class for GRPO training.
Tracks metrics, generates plots, and monitors memory usage.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import torch
from config import config


class TrainingMonitor:
    """Monitors training progress, logs metrics, and generates visualizations."""
    
    def __init__(self):
        """Initialize training monitor with empty metrics storage."""
        self.metrics = {
            'rewards': [],
            'losses': [],
            'accuracies': [],
            'steps': []
        }
        self.start_time = datetime.now()
        
        # Create logs directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
    
    def log_metrics(self, metrics_dict, step):
        """Log training metrics and perform periodic saving/plotting.
        
        Args:
            metrics_dict (dict): Dictionary containing metric values
            step (int): Current training step
        """
        # Store metrics
        if 'reward' in metrics_dict:
            self.metrics['rewards'].append(metrics_dict['reward'])
            self.metrics['steps'].append(step)
        
        # Console output
        print(f"📊 Step {step}: Reward={metrics_dict.get('reward', 'N/A'):.3f}")
        
        # Periodic saving
        if step % config.logging_steps == 0:
            self._save_metrics()
            self.plot_training_progress()
            self.log_memory_usage()  # Memory monitoring
    
    def _save_metrics(self):
        """Save metrics to JSON file for persistence."""
        metrics_path = os.path.join(config.output_dir, 'training_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_training_progress(self):
        """Generate training progress visualization plots."""
        if len(self.metrics['rewards']) < 2:
            return  # Not enough data
        
        plt.figure(figsize=(15, 5))
        
        # Reward evolution plot
        plt.subplot(131)
        plt.plot(self.metrics['steps'], self.metrics['rewards'], 'b-', alpha=0.7)
        plt.title('Reward Evolution')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Moving average reward (10-step window)
        if len(self.metrics['rewards']) > 10:
            plt.subplot(132)
            moving_avg = np.convolve(self.metrics['rewards'], np.ones(10)/10, mode='valid')
            plt.plot(self.metrics['steps'][9:], moving_avg, 'r-', linewidth=2)
            plt.title('Moving Average Reward (10 steps)')
            plt.xlabel('Step')
            plt.ylabel('Moving Average Reward')
            plt.grid(True, alpha=0.3)
        
        # Reward distribution histogram
        plt.subplot(133)
        plt.hist(self.metrics['rewards'], bins=20, alpha=0.7, edgecolor='black', color='green')
        plt.title('Reward Distribution')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(config.output_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_path}")
    
    def log_memory_usage(self):
        """Log GPU memory usage and check 16GB VRAM constraint."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"GPU Memory:")
            print(f"  - Current: {allocated:.2f}GB")
            print(f"  - Reserved: {reserved:.2f}GB") 
            print(f"  - Max: {max_allocated:.2f}GB")
            
            # Check 16GB VRAM constraint
            if max_allocated > 16:
                print("⚠️ WARNING: Exceeding 16GB VRAM limit!")
            else:
                print(f"Within limit: {max_allocated:.2f}GB / 16GB")
        else:
            print("CPU mode - no GPU memory monitoring")
    
    def final_report(self):
        """Generate final training report with statistics and metrics."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("FINAL TRAINING REPORT - BNP PARIBAS TEST")
        print("="*60)
        print(f"⏱️  Total duration: {total_time/60:.1f} minutes")
        print(f"Completed steps: {len(self.metrics['steps'])}")
        
        if self.metrics['rewards']:
            avg_reward = np.mean(self.metrics['rewards'])
            max_reward = np.max(self.metrics['rewards'])
            std_reward = np.std(self.metrics['rewards'])
            
            print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"Maximum reward: {max_reward:.3f}")
            
            # Calculate correct answer percentage
            correct_answers = sum(1 for r in self.metrics['rewards'] if r > 0.5)
            accuracy = correct_answers / len(self.metrics['rewards']) if self.metrics['rewards'] else 0
            print(f"Accuracy: {accuracy:.2%} ({correct_answers}/{len(self.metrics['rewards'])})")
        
        # Final memory report
        self.log_memory_usage()
        
        # Final save
        self._save_metrics()
        self.plot_training_progress()
        
        print(f"Data saved in: {config.output_dir}")
        print("="*60)


# Testing the monitoring system
if __name__ == "__main__":
    print("🧪 Testing monitoring system...")
    monitor = TrainingMonitor()
    
    # Test with simulated data
    for i in range(50):
        mock_metrics = {'reward': np.random.uniform(-0.5, 1.0)}
        monitor.log_metrics(mock_metrics, i)
    
    monitor.final_report()
    print("Monitoring test completed successfully!")