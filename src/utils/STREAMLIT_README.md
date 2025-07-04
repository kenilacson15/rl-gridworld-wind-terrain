# 🎮 GridWorld RL Streamlit Dashboard

A comprehensive interactive visualization dashboard for the GridWorld Reinforcement Learning project, built with Streamlit and Plotly.

## ✨ Features

### 🎯 **Interactive Environment Visualization**
- **Real-time GridWorld rendering** with Plotly interactive plots
- **Terrain visualization** - Ice ❄️, Mud 🟤, Quicksand 🏜️
- **Wind zones** with directional indicators and strength
- **Agent trajectory tracking** with live path visualization
- **Value function heatmaps** for Q-Learning/SARSA agents
- **Policy visualization** showing optimal actions per state

### 📊 **Live Training Metrics**
- **Real-time plots** for rewards, steps, success rate, epsilon decay
- **Rolling averages** for smoothed trend analysis
- **Performance statistics** with key metrics
- **Learning curve analysis** with multiple window sizes

### 🛠️ **Interactive Configuration**
- **Environment settings** - Grid size, goal position
- **Agent selection** - Q-Learning, SARSA, DQN, Value/Policy Iteration
- **Terrain configuration** - Interactive terrain placement and properties
- **Wind zone setup** - Configurable wind directions and strengths
- **Training parameters** - Learning rate, epsilon, episodes

### 🔍 **Advanced Analysis Tools**
- **Environment analysis** - Configuration summaries and statistics
- **Agent analysis** - Q-table visualization, policy inspection
- **Performance analysis** - Learning curves, statistical summaries
- **Data export** - Download training data as CSV

## 🚀 Quick Start

### 1. Install Requirements
```bash
# Option 1: Install from root directory
pip install -r streamlit_requirements.txt

# Option 2: Install from utils directory  
pip install -r src/utils/streamlit_requirements.txt

# Option 3: Install individually
pip install streamlit plotly pandas
```

### 2. Launch Dashboard
```bash
# Option 1: Use the launcher script (recommended)
python launch_dashboard.py

# Option 2: Direct Streamlit command
streamlit run src/utils/streamlit_viz.py
```

### 3. Access Dashboard
- Dashboard opens automatically in your browser
- Default URL: `http://localhost:8501`
- Mobile-friendly responsive design

## 🎮 How to Use

### **Environment Setup**
1. **Configure Grid Size** - Use sidebar slider (5x5 to 15x15)
2. **Set Goal Position** - Adjust goal coordinates
3. **Add Terrain** - Enable/disable terrain types and set positions
4. **Configure Wind** - Set wind zones with direction and strength

### **Training Configuration**
1. **Select Agent** - Choose from Q-Learning, SARSA, DQN, etc.
2. **Set Parameters** - Learning rate, epsilon, number of episodes
3. **Start Training** - Click "🚀 Start Training"
4. **Monitor Progress** - Watch live metrics and visualization

### **Analysis Features**
1. **Environment Tab** - View configuration and terrain statistics
2. **Agent Tab** - Inspect Q-tables and policies (for tabular methods)
3. **Performance Tab** - Analyze learning curves and statistics

## 📊 Dashboard Components

### **Main Visualization**
- **Interactive GridWorld** - Plotly-based environment rendering
- **Live Agent Tracking** - Real-time position updates
- **Terrain Overlays** - Visual terrain effects
- **Value Heatmaps** - Agent's learned value function

### **Sidebar Controls**
- **Environment Configuration** - Grid, goal, terrain, wind settings
- **Agent Settings** - Algorithm selection and hyperparameters
- **Training Controls** - Start, stop, reset, export buttons

### **Metrics Dashboard**
- **Real-time Plots** - Rewards, steps, success rate over episodes
- **Rolling Averages** - Smoothed trend visualization
- **Current Statistics** - Live performance metrics

### **Analysis Tools**
- **Q-Table Inspector** - Interactive Q-value exploration
- **Policy Visualization** - Heatmap of optimal actions
- **Learning Curves** - Multi-window trend analysis
- **Data Export** - CSV download functionality

## 🎯 Use Cases

### **Research & Development**
- **Algorithm Comparison** - Side-by-side agent performance
- **Hyperparameter Tuning** - Interactive parameter adjustment
- **Environment Design** - Visual terrain and wind configuration
- **Behavior Analysis** - Q-table and policy inspection

### **Education & Demonstration**
- **RL Concept Visualization** - Live learning process
- **Interactive Experiments** - Real-time parameter effects
- **Progress Tracking** - Clear metric visualization
- **Algorithm Understanding** - Policy and value function display

### **Prototyping & Testing**
- **Rapid Experimentation** - Quick configuration changes
- **Visual Debugging** - Environment and agent behavior inspection
- **Performance Monitoring** - Live metric tracking
- **Data Collection** - Training data export for analysis

## 🔧 Technical Details

### **Architecture**
- **Frontend**: Streamlit with Plotly visualization
- **Backend**: Python with NumPy/Pandas for data processing
- **Visualization**: Interactive Plotly charts and heatmaps
- **State Management**: Streamlit session state for training persistence

### **Performance Optimizations**
- **Efficient Rendering** - Optimized Plotly figure updates
- **Memory Management** - Limited history storage
- **Real-time Updates** - Smart refresh scheduling
- **Responsive Design** - Adaptive layout for different screen sizes

### **Data Export Format**
```csv
episode,reward,steps,success,epsilon,timestamp
0,-45.0,78,0,0.995,2024-01-01T12:00:00
1,-23.0,56,1,0.990,2024-01-01T12:01:00
...
```

## 🚨 Troubleshooting

### **Common Issues**

#### Import Errors
```bash
# Make sure you're in the project root
cd /path/to/rl-gridworld-wind-terrain

# Install missing packages
pip install streamlit plotly pandas
```

#### Port Already in Use
```bash
# Use different port
streamlit run src/utils/streamlit_viz.py --server.port 8502
```

#### Slow Performance
- Reduce grid size for faster rendering
- Lower update frequency in training loop
- Close other browser tabs to free memory

### **Browser Compatibility**
- **Recommended**: Chrome, Firefox, Safari
- **Mobile**: Responsive design works on tablets/phones
- **JavaScript**: Required for interactive features

## 📈 Performance Tips

### **For Large Grids (10x10+)**
- Use lower update frequencies
- Limit trajectory history
- Reduce value heatmap resolution

### **For Long Training Sessions**
- Export data periodically
- Monitor memory usage
- Use rolling windows for metrics

### **For Real-time Analysis**
- Keep browser tab active
- Close unnecessary applications
- Use fast refresh rates only when needed

## 🔮 Future Enhancements

- **Multi-agent support** - Multiple agents in same environment
- **3D visualization** - Enhanced environment rendering
- **Video export** - Training session recordings
- **Custom environments** - User-defined environment types
- **Advanced analysis** - Statistical testing, confidence intervals
- **Collaboration features** - Shared sessions and configurations

## 📝 Notes

- **Session Persistence**: Training state persists during browser session
- **Data Export**: CSV format compatible with pandas/Excel
- **Real-time Updates**: Dashboard refreshes automatically during training
- **Mobile Support**: Responsive design works on tablets and phones

---

**Enjoy exploring reinforcement learning with the interactive GridWorld dashboard!** 🎮📊
