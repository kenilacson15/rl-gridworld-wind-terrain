import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.gridworld import GridWorldEnv
import numpy as np
import pytest 



def test_reset_returns_start_position():
    env = GridWorldEnv()
    state = env.reset()
    assert isinstance(state, np.ndarray), "State should be a numpy array."
    assert (state == np.array(env.config["start_pos"])).all()


def test_step_output_format():
    env = GridWorldEnv()
    state = env.reset()
    next_state, reward, done, info = env.step(1)
    assert isinstance(next_state, np.ndarray), "next_state should be a numpy array."
    assert isinstance(reward, (int, float)), "Reward should be numeric."
    assert isinstance(done, bool), "Done should be boolean."
    assert "position" in info, "Info should be contain 'position'."



def test_boundary_behavior():
    env = GridWorldEnv()
    state = env.reset()
    state, _, _, _ = env.step(3)
    assert (state == np.array(env.config["start_pos"])).all()

def test_wind_effect():
    config = {"grid_size": (5, 5), "wind": [0, 0, 1, 0, 0], "start_pos": (2, 2), "goal_pos": (4, 4)}
    env = GridWorldEnv(config)
    state = env.reset()
    next_state, _, _, _ = env.step(1)
    assert next_state[0] <= state[0], "Agent should be pushed upwards."



def test_ice_slip_effect(monkeypatch):
    env = GridWorldEnv()
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "choice", lambda x: 0)
    state = env.reset()
    next_state, _, _, _ = env.step(1)


def test_goal_reached():
    config = {"grid_size": (5, 5), "wind": [0, 0, 0, 0, 0], "stochastic_terrain": {}, "goal_pos": (0, 1), "start_pos": (0, 0)}
    env = GridWorldEnv(config)
    state = env.reset()
    next_state, reward, done, _ = env.step(1)
    assert (next_state == np.array([0, 1])).all()
    assert reward == 0
    assert done is True
