# GridWorld with Wind and Stochastic Terrain

A customizable 2D Reinforcement Learning environment where an agent learns to navigate through wind zones and stochastic terrain patches.

## 📖 Overview

This project provides a fully featured GridWorld environment designed to help researchers and learners explore core concepts of Reinforcement Learning (RL), such as Markov Decision Processes, Value Iteration, and Policy Iteration. Its modular design allows easy extension for advanced techniques (e.g., DQN, SARSA) and custom environment dynamics.

## ✨ Features

* **Stochastic Terrain:** Includes ice and swamp patches.
* **Wind Effects:** Forces the agent to adjust its path.
* **Configurable Grid:** Flexible layout and terrain probabilities.
* **Agents:** Q-Learning, Value Iteration, and Policy Iteration (with examples).
* **Visualization Tools:** Heatmaps, trajectory plotting, and metric dashboards.
* **Modular Architecture:** Enables clean separation of environments, agents, and utilities.

## 🛠️ Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:

   ```bash
   python src/main.py
   ```

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
