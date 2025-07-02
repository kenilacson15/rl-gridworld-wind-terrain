# GridWorld with Wind and Stochastic Terrain

A customizable 2D Reinforcement Learning environment where an agent learns to navigate through wind zones and stochastic terrain patches.

## 📖 Overview

This project provides a fully featured GridWorld environment designed to help researchers and learners explore core concepts of Reinforcement Learning (RL), such as Markov Decision Processes, Value Iteration, and Policy Iteration. Its modular design allows easy extension for advanced techniques (e.g., DQN, SARSA) and custom environment dynamics.

## ✨ Features

* **Stochastic Terrain:** Includes ice and swamp patches.
* **Wind Effects:** Forces the agent to adjust its path.
* **Configurable Grid:** Flexible layout and terrain probabilities.
* **Agents:** Q-Learning, Value Iteration, and Policy Iteration (with examples).
* **Visualization Tools:** 
  * PyGame interactive visualization with real-time agent behavior
  * Heatmaps, trajectory plotting, and metric dashboards
* **Modular Architecture:** Enables clean separation of environments, agents, and utilities.

## 🛠️ Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script with PyGame visualization (default):

   ```bash
   python src/main.py
   ```

3. Command-line options:

   ```bash
   # Run with a specific agent
   python src/main.py --agent q_learning

   # Run with SARSA agent for 100 episodes
   python src/main.py --agent sarsa --episodes 100

   # Disable PyGame visualization
   python src/main.py --no-pygame
   ```

## 🎮 Enhanced Interactive Visualization

The project includes an enhanced PyGame-based visualization with textures, animations, and a modern UI:

- **Textured Terrain**: Different terrains (ice, mud, quicksand) rendered with detailed textures
- **Animated Agent**: Smooth agent movement with directional sprites
- **Particle Effects**: Dynamic wind particles and goal sparkles
- **Enhanced Visuals**: Gradient backgrounds, glowing elements, and pulsing effects
- **Modern UI Panel**: Stylish information panel with episode stats and controls
- **Advanced Heatmap**: Multi-color gradient visualization for state values
- **Animated Policy Arrows**: Dynamic policy indicators showing agent decisions
- **Optimized Performance**: Efficient rendering using dirty rectangles and frame rate control

### Running the Enhanced Visualization:

```bash
# Run the visualization test script
python src/run_visual_test.py

# Specify agent type (q_learning, dqn, sarsa, random)
python src/run_visual_test.py --agent q_learning

# Train the agent before visualization
python src/run_visual_test.py --agent dqn --train --train-steps 2000

# Force regeneration of textures and fonts
python src/run_visual_test.py --generate-assets

# Set window size
python src/run_visual_test.py --window-size 1280x800
```

### Controls:

- **Space:** Pause/resume simulation
- **V:** Toggle value function visualization
- **P:** Toggle policy arrows
- **+/-:** Adjust animation speed
- **Q/ESC:** Quit visualization

## 📁 Project Structure

```
gridworld-wind-terrain/
├─ data/
├─ models/
├─ notebooks/
├─ experiments/
├─ src/
├─ reports/
├─ tests/
├─ .gitignore
├─ README.md
├─ CONTRIBUTING.md
├─ requirements.txt
└─ LICENSE
```

## 📋 Project Status (as of June 2025)

✅ GridWorld environment (wind, terrain, stochastic dynamics)
✅ Q-Learning, Value Iteration, and Policy Iteration implemented
✅ Visualization tools and metrics plotting
☑️ ValueIterationAgent review required
☑️ Advanced multi-agent and deep RL methods planned

## 🗺️ Roadmap

* ✅ Phase 1: Environment design & basic dynamics
* ✅ Phase 2: Agent implementation and benchmarking
* ✅ Phase 3: Experimentation, metrics, and plotting
* ✅ Phase 4: Scalability and extensibility (POMDP, multi-agent)

## 💻 Contributing

We welcome contributions! Please review the [CONTRIBUTING.md](CONTRIBUTING.md) for details on submitting PRs and running tests.

## ⚡️ Key Dependencies

* `numpy`, `matplotlib`, `tqdm`, `gym`, `pytest`, `pandas`, `seaborn`
* Optional: `mlflow`, `hydra-core`, `dvc`, `black`, `isort`, `flake8`

## 📄 License

Open Source (see [LICENSE](LICENSE)).

## 📞 Contact

For questions or collaboration, open an issue or contact the maintainer.
