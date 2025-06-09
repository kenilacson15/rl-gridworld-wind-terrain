# 🌪️ GridWorld with Wind and Stochastic Terrain

A custom 2D reinforcement learning environment for exploring core RL concepts such as value iteration, policy learning, and stochastic transitions. This project simulates dynamic terrains and wind zones to create a challenging, educational environment for RL experimentation.

---

## 📌 Project Goal

Build an original, beginner-friendly RL environment to help solidify foundational concepts:

- Markov Decision Processes (MDPs)
- Value and Policy Iteration
- Transition Probability Modeling
- Exploration vs. Exploitation

---

## ✨ Features

- ✅ **Grid-based environment**: Agent navigation in a customizable grid
- 🌬️ **Wind zones**: Affect agent movement direction
- 🧊 **Terrain types**: Ice (slippery), swamp (slow/sticky), and more
- 🎲 **Stochastic & deterministic dynamics**: Realistic RL challenges
- 📊 **Classical planning algorithms**: Value iteration, policy iteration

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```powershell
   git clone <repo-url>
   cd rl-gridworld-wind-terrain
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run a sample experiment:**
   ```powershell
   python src/main.py
   ```

---

## 📁 Project Structure

```bash
gridworld-rl/
│
├── src/                # Environment logic, agent models
│   ├── envs/           # Environment classes
│   ├── agents/         # Agent implementations
│   ├── utils/          # Utility functions
│   └── main.py         # Entry point
│
├── notebooks/          # Experimentation & visualization
├── data/               # Raw and processed data
│   ├── raw/
│   └── processed/
├── models/             # Saved models and logs
│   ├── trained/
│   └── logs/
├── reports/            # Generated analysis and reports
├── tests/              # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## 🏁 Getting Started

- Check out the `notebooks/` folder for interactive demos.
- Modify `src/envs/` to create custom environments.
- Use `src/agents/` to implement new RL agents.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or improvements.

---

## 📄 License

[MIT License](LICENSE)
