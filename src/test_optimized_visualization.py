"""
Test script for the optimized Pygame/Matplotlib integration.

This script demonstrates the smooth integration of Pygame and Matplotlib
visualizations using the async_visualization module.
"""

import os
import sys
import time
import random
import numpy as np
# Configure Matplotlib backend before importing plt
import matplotlib
matplotlib.use('TkAgg')  # TkAgg works well with threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
# Import pygame constants explicitly
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

# Add parent directory to path to ensure imports work properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG
from utils.game_visual import GridWorldVisualizer
from utils.async_visualization import AsyncMetricsVisualizer, VisualizationCoordinator


def test_optimized_visualization():
    """Run a test of the optimized visualization."""
    # Initialize pygame early to avoid conflicts with matplotlib
    pygame.init()
    
    # Create visualization coordinator
    viz_coordinator = VisualizationCoordinator(
        pygame_fps=30.0,  # Limit Pygame to 30 FPS
        metrics_update_interval=0.5  # Update metrics every 0.5 seconds
    )
    
    # Create environment and agent
    env = GridWorldEnv(DEFAULT_ENV_CONFIG)
    agent = QLearningAgent(env, QL_AGENT_CONFIG)
    
    # Create visualizers
    pygame_vis = GridWorldVisualizer()
    
    # Initialize metrics
    metrics = {"rewards": [], "steps": [], "successes": []}
    
    # Create metrics visualizer after pygame is initialized
    metrics_vis = viz_coordinator.create_metrics_visualizer("Q-Learning")
    
    # Ensure matplotlib is ready for threading
    plt.ion()  # Turn on interactive mode
    
    # Training loop
    num_episodes = 20
    goal = tuple(env.config["goal_pos"])
    running = True
    
    try:
        for ep in range(num_episodes):
            if not running:
                break
                
            obs = env.reset()
            total_reward, done, steps = 0, False, 0
            
            print(f"Episode {ep+1}/{num_episodes}")
            
            while not done and running:
                # Handle pygame events - use a time limit to avoid blocking
                start_event_time = time.time()
                # Process a limited number of events per frame to avoid starvation
                for _ in range(10):  # Process up to 10 events
                    if not pygame.event.get_busy():
                        event = pygame.event.poll()
                        if event.type == QUIT:
                            running = False
                            break
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE:
                                running = False
                                break
                    # Don't spend more than 5ms processing events
                    if time.time() - start_event_time > 0.005:
                        break
                
                # Agent step
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.update(obs, action, reward, next_obs, None if done else agent.act(next_obs))
                obs = next_obs
                total_reward += reward
                steps += 1
                
                # Update PyGame visualization with rate limiting
                if viz_coordinator.should_update_pygame():
                    try:
                        pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)
                    except Exception as e:
                        print(f"Error in pygame rendering: {e}")
                
                # Give the main thread a short break to allow matplotlib to update
                # This helps prevent thread starvation
                time.sleep(0.001)
            
            # Record metrics at episode end
            is_success = tuple(env.agent_pos) == goal
            metrics["rewards"].append(total_reward)
            metrics["steps"].append(steps)
            metrics["successes"].append(1 if is_success else 0)
            
            # Update metrics visualization (non-blocking)
            try:
                metrics_vis.update(metrics)
                # Process any pending matplotlib events
                plt.pause(0.001)  # Short pause to let matplotlib process events
            except Exception as e:
                print(f"Error updating metrics: {e}")
            
            # Print episode summary
            print(f"Episode {ep+1}: Reward={total_reward:.1f}, Steps={steps}, Success={is_success}")
    
    finally:
        # Cleanup in reverse order of creation
        print("Cleaning up resources...")
        try:
            # Close matplotlib figures first
            plt.close('all')
            
            # Then cleanup the visualization coordinator
            if viz_coordinator:
                viz_coordinator.cleanup()
            
            # Finally quit pygame
            pygame.quit()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        print("Cleanup complete")


if __name__ == "__main__":
    try:
        test_optimized_visualization()
    except Exception as e:
        print(f"Error in visualization test: {e}")
        import traceback
        traceback.print_exc()
        # Make sure pygame quits even on error
        pygame.quit()
