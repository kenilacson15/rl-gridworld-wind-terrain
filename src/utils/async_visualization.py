"""
Async Visualization Module

This module provides utilities for asynchronous visualization updates
to ensure both Pygame and Matplotlib can run smoothly together without
blocking each other's rendering loops.
"""

import threading
import queue
import time
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import os


class AsyncMetricsVisualizer:
    """
    Asynchronous metrics visualizer that updates plots in a separate thread
    to avoid blocking the main game loop.
    """
    
    def __init__(self, agent_type: str, update_interval: float = 0.5):
        """
        Initialize the async metrics visualizer.
        
        Args:
            agent_type: Type of agent ('q_learning', 'sarsa', 'dqn', etc.)
            update_interval: Minimum time (seconds) between plot updates
        """
        self.agent_type = agent_type.upper()
        self.update_interval = update_interval
        self.last_update_time = 0
        
        # Create a thread-safe queue for metrics updates
        self.metrics_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Create the figure and axes (will be used in the thread)
        # Use interactive mode for the main thread
        plt.ion()
        
        # Create a figure with subplot for metrics visualization
        self.fig, (self.r_ax, self.s_ax, self.succ_ax) = plt.subplots(1, 3, figsize=(15, 4))
        self.setup_axes()
        
        # Initialize latest metrics
        self.latest_metrics = {"rewards": [], "steps": [], "successes": []}
        
        # Create a lock for thread-safe operations
        self.lock = threading.RLock()
        
        # Start the update thread
        self.start()
    
    def setup_axes(self):
        """Set up the axes with proper labels and titles."""
        self.fig.suptitle(f"{self.agent_type} Training Metrics")
        
        self.r_ax.set_title("Rewards per Episode")
        self.r_ax.set_xlabel("Episode")
        self.r_ax.set_ylabel("Total Reward")
        
        self.s_ax.set_title("Steps per Episode")
        self.s_ax.set_xlabel("Episode")
        self.s_ax.set_ylabel("Steps")
        
        self.succ_ax.set_title("Success Rate")
        self.succ_ax.set_xlabel("Episode")
        self.succ_ax.set_ylabel("Success Rate (10-ep avg)")
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def start(self):
        """Start the update thread if not already running."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._update_thread, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the update thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.0)  # Wait for thread to finish
            except Exception as e:
                print(f"Error stopping visualization thread: {e}")
    
    def update(self, metrics: Dict[str, List]):
        """
        Queue metrics for update. This method is thread-safe and non-blocking.
        
        Args:
            metrics: Dictionary containing 'rewards', 'steps', and 'successes' lists
        """
        try:
            # Create a deep copy of the metrics to avoid modification while plotting
            metrics_copy = {
                "rewards": metrics.get("rewards", []).copy(),
                "steps": metrics.get("steps", []).copy(),
                "successes": metrics.get("successes", []).copy()
            }
            
            # Add metrics to the update queue
            self.metrics_queue.put(metrics_copy)
        except Exception as e:
            print(f"Error queuing metrics update: {e}")
    
    def _update_thread(self):
        """Thread function that processes the metrics queue and updates plots."""
        while self.running:
            try:
                # Get the latest metrics from the queue (non-blocking)
                try:
                    # Get all available metrics, keeping only the most recent
                    while not self.metrics_queue.empty():
                        self.latest_metrics = self.metrics_queue.get_nowait()
                        self.metrics_queue.task_done()
                except queue.Empty:
                    pass
                
                # Check if it's time to update the plots
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    self._update_plots()
                    self.last_update_time = current_time
                
                # Sleep briefly to avoid consuming too much CPU
                time.sleep(0.05)
            except Exception as e:
                print(f"Error in visualization thread: {e}")
                time.sleep(0.5)  # Sleep longer on error
    
    def _update_plots(self):
        """Update the plots with the latest metrics."""
        try:
            with self.lock:
                # Check if the figure still exists
                if not plt.fignum_exists(self.fig.number):
                    # The figure was closed, recreate it
                    self.fig, (self.r_ax, self.s_ax, self.succ_ax) = plt.subplots(1, 3, figsize=(15, 4))
                    self.setup_axes()
                
                # Clear axes for new data
                self.r_ax.clear()
                self.s_ax.clear()
                self.succ_ax.clear()
                
                # Plot rewards
                self.r_ax.plot(self.latest_metrics["rewards"], 'b-')
                self.r_ax.grid(True)
                
                # Plot steps
                self.s_ax.plot(self.latest_metrics["steps"], 'g-')
                self.s_ax.grid(True)
                
                # Plot success rate
                if self.latest_metrics["successes"]:
                    window = min(10, len(self.latest_metrics["successes"]))
                    rate = np.convolve(self.latest_metrics["successes"], 
                                    np.ones(window)/window, mode='valid')
                    self.succ_ax.plot(rate, 'r-')
                    self.succ_ax.set_ylim(-0.1, 1.1)
                self.succ_ax.grid(True)
                
                # Reset axis labels and titles (they get cleared with clear())
                self.setup_axes()
                
                # Use a thread-safe approach to update the figure
                try:
                    # Draw the plot without displaying immediately
                    self.fig.canvas.draw_idle()
                    
                    # Attempt to flush events if we're in the main thread
                    if threading.current_thread() is threading.main_thread():
                        self.fig.canvas.flush_events()
                except Exception as e:
                    # Fallback for non-main thread: save to a temporary file and reload
                    temp_file = os.path.join(os.path.dirname(__file__), "temp_plot.png")
                    self.fig.savefig(temp_file)
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def save(self, filepath: str):
        """Save the current figure to a file."""
        with self.lock:
            try:
                self.fig.savefig(filepath, bbox_inches='tight', dpi=300)
            except Exception as e:
                print(f"Error saving figure to {filepath}: {e}")


class UpdateRateLimiter:
    """
    Rate limiter for rendering updates to prevent overwhelming CPU/GPU.
    """
    
    def __init__(self, max_fps: float = 30.0):
        """
        Initialize the rate limiter.
        
        Args:
            max_fps: Maximum frames per second
        """
        self.min_interval = 1.0 / max_fps
        self.last_update_time = 0
    
    def should_update(self) -> bool:
        """
        Check if enough time has passed to allow another update.
        
        Returns:
            True if an update should be performed, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.min_interval:
            self.last_update_time = current_time
            return True
        return False


class VisualizationCoordinator:
    """
    Coordinates updates between Pygame and Matplotlib visualizations to
    ensure smooth performance.
    """
    
    def __init__(self, pygame_fps: float = 30.0, metrics_update_interval: float = 0.5):
        """
        Initialize the visualization coordinator.
        
        Args:
            pygame_fps: Maximum frames per second for Pygame updates
            metrics_update_interval: Seconds between metrics plot updates
        """
        self.pygame_limiter = UpdateRateLimiter(max_fps=pygame_fps)
        self.metrics_update_interval = metrics_update_interval
        self.async_visualizer = None
        
        # Set matplotlib backend for best performance with pygame
        try:
            current_backend = matplotlib.get_backend()
            if 'TkAgg' not in current_backend and 'Qt' not in current_backend:
                plt.switch_backend('TkAgg')  # TkAgg works well with pygame
        except Exception as e:
            print(f"Warning: Could not switch matplotlib backend: {e}")
    
    def create_metrics_visualizer(self, agent_type: str) -> AsyncMetricsVisualizer:
        """
        Create and return an async metrics visualizer.
        
        Args:
            agent_type: Type of agent ('q_learning', 'sarsa', 'dqn', etc.)
        
        Returns:
            AsyncMetricsVisualizer instance
        """
        self.async_visualizer = AsyncMetricsVisualizer(
            agent_type=agent_type,
            update_interval=self.metrics_update_interval
        )
        return self.async_visualizer
    
    def should_update_pygame(self) -> bool:
        """
        Check if a Pygame update should be performed.
        
        Returns:
            True if a Pygame update should occur, False otherwise
        """
        return self.pygame_limiter.should_update()
    
    def cleanup(self):
        """
        Clean up resources, stop threads, etc.
        """
        if self.async_visualizer:
            self.async_visualizer.stop()
