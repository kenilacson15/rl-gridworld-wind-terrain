"""
Metrics logging utilities for reinforcement learning experiments.
This module provides functions to save training metrics to CSV files
and generate enhanced visualizations with detailed information.
"""

import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional


def get_timestamp_str() -> str:
    """Generate a timestamp string for filenames.
    
    Returns:
        str: Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_metrics_to_csv(metrics: Dict[str, List[Any]], agent_type: str, 
                        logs_dir: str = 'models/logs') -> str:
    """Save training metrics to a CSV file.
    
    Args:
        metrics: Dictionary containing lists of rewards, steps, and successes
        agent_type: Type of agent used for training (q_learning, sarsa, dqn)
        logs_dir: Directory to save logs to
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a DataFrame from metrics
    df = pd.DataFrame({
        'episode': range(len(metrics['rewards'])),
        'reward': metrics['rewards'],
        'steps': metrics['steps'],
        'success': metrics['successes']
    })
    
    # Add timestamp to filename
    timestamp = get_timestamp_str()
    csv_filename = f"{agent_type}_metrics_{timestamp}.csv"
    csv_path = os.path.join(logs_dir, csv_filename)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    print(f"📝 Training metrics saved to: {csv_path}")
    return csv_path


def generate_enhanced_plot(metrics: Dict[str, List[Any]], agent_type: str, 
                          logs_dir: str = 'models/logs') -> str:
    """Generate an enhanced metrics plot with detailed information.
    
    Args:
        metrics: Dictionary containing lists of rewards, steps, and successes
        agent_type: Type of agent used for training (q_learning, sarsa, dqn)
        logs_dir: Directory to save the plot to
        
    Returns:
        str: Path to the saved plot file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Calculate statistics
    total_episodes = len(metrics['rewards'])
    avg_reward = np.mean(metrics['rewards'])
    avg_steps = np.mean(metrics['steps'])
    success_rate = np.mean(metrics['successes'])
    
    # Create figure with 3 subplots
    fig, (r_ax, s_ax, succ_ax) = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    # Add main title with detailed information
    fig.suptitle(
        f"{agent_type.upper()} Training Results\n"
        f"Total Episodes: {total_episodes} | Avg Reward: {avg_reward:.2f} | "
        f"Avg Steps: {avg_steps:.2f} | Success Rate: {success_rate:.2%}",
        fontsize=14, fontweight='bold'
    )
    
    # Plot rewards
    if metrics["rewards"]:
        episodes = list(range(len(metrics["rewards"])))
        r_ax.plot(episodes, metrics["rewards"], 'b-', linewidth=1.5)
        
        # Add rolling average line
        window = min(10, len(metrics["rewards"]))
        if window > 0:
            rolling_avg = np.convolve(metrics["rewards"], 
                                    np.ones(window)/window, mode='valid')
            r_ax.plot(range(window-1, len(metrics["rewards"])), 
                     rolling_avg, 'r-', linewidth=1.5, alpha=0.7,
                     label=f'{window}-ep Rolling Avg')
        
        r_ax.set_title(f"Rewards per Episode\nAvg: {avg_reward:.2f}", fontsize=12)
        r_ax.set_xlabel("Episode", fontsize=10)
        r_ax.set_ylabel("Total Reward", fontsize=10)
        r_ax.grid(True, alpha=0.3)
        r_ax.legend()
    
    # Plot steps
    if metrics["steps"]:
        episodes = list(range(len(metrics["steps"])))
        s_ax.plot(episodes, metrics["steps"], 'g-', linewidth=1.5)
        
        # Add rolling average line
        window = min(10, len(metrics["steps"]))
        if window > 0:
            rolling_avg = np.convolve(metrics["steps"], 
                                    np.ones(window)/window, mode='valid')
            s_ax.plot(range(window-1, len(metrics["steps"])), 
                     rolling_avg, 'r-', linewidth=1.5, alpha=0.7,
                     label=f'{window}-ep Rolling Avg')
        
        s_ax.set_title(f"Steps per Episode\nAvg: {avg_steps:.2f}", fontsize=12)
        s_ax.set_xlabel("Episode", fontsize=10)
        s_ax.set_ylabel("Steps", fontsize=10)
        s_ax.grid(True, alpha=0.3)
        s_ax.legend()
    
    # Plot success rate
    if metrics["successes"]:
        window = min(10, len(metrics["successes"]))
        if window > 0:
            rate = np.convolve(metrics["successes"], 
                             np.ones(window)/window, mode='valid')
            episodes = list(range(len(rate)))
            succ_ax.plot(episodes, rate, 'r-', linewidth=1.5)
            succ_ax.set_title(f"Success Rate\nFinal: {success_rate:.2%}", fontsize=12)
            succ_ax.set_xlabel("Episode", fontsize=10)
            succ_ax.set_ylabel("Success Rate (10-ep avg)", fontsize=10)
            succ_ax.set_ylim(-0.1, 1.1)
            succ_ax.grid(True, alpha=0.3)
    
    # Add timestamp and final episode metrics to the bottom
    plt.figtext(
        0.5, 0.01, 
        f"Final 10 Episodes - Avg Reward: {np.mean(metrics['rewards'][-10:]):.2f} | "
        f"Success Rate: {np.mean(metrics['successes'][-10:]):.2%}",
        ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
    )
    
    # Add timestamp to filename
    timestamp = get_timestamp_str()
    plot_filename = f"{agent_type}_plot_{timestamp}.png"
    plot_path = os.path.join(logs_dir, plot_filename)
    
    # Save the figure
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"📊 Enhanced metrics plot saved to: {plot_path}")
    return plot_path
